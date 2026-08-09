#!/usr/bin/env node
/**
 * Entry 021 Step 06 only — RAG citation UI debug.
 * API: SSE meta/done sources. UI: chips immediately after stream vs after refresh.
 */
import { chromium } from '../frontend/node_modules/playwright/index.mjs'
import fs from 'fs'
import path from 'path'
import { fileURLToPath } from 'url'

const __dirname = path.dirname(fileURLToPath(import.meta.url))
const ROOT = path.resolve(__dirname, '..')
const API = 'http://127.0.0.1:8000'
const APP = 'http://127.0.0.1:5173'
const RAG_MSG =
  'According to the Dietary Guidelines for Americans, how many vegetables should adults eat each day, and what types are recommended?'
const NORMAL_MSG = 'What is one practical tip to stay consistent with healthy eating this week?'

async function api(method, route, body) {
  const res = await fetch(`${API}${route}`, {
    method,
    headers: { 'Content-Type': 'application/json' },
    body: body ? JSON.stringify(body) : undefined,
  })
  if (!res.ok) throw new Error(`${method} ${route} → ${res.status}`)
  return res.json()
}

async function parseSseStream(uid, message) {
  const res = await fetch(`${API}/chat/stream`, {
    method: 'POST',
    headers: { 'Content-Type': 'application/json' },
    body: JSON.stringify({ user_id: uid, message }),
  })
  const events = { meta_sources: [], done_sources: [], token_count: 0 }
  const text = await res.text()
  for (const block of text.split('\n\n')) {
    if (!block.trim()) continue
    let ev = 'message'
    let data = null
    for (const line of block.split('\n')) {
      if (line.startsWith('event:')) ev = line.slice(6).trim()
      if (line.startsWith('data:')) data = JSON.parse(line.slice(5))
    }
    if (ev === 'meta') events.meta_sources = data?.sources || []
    if (ev === 'token') events.token_count += 1
    if (ev === 'done') events.done_sources = data?.sources || []
  }
  return events
}

async function chipState(page) {
  const last = page.locator('.message-row.assistant').last()
  const count = await last.locator('.source-chip').count()
  const texts = count ? await last.locator('.source-chip').allTextContents() : []
  return { count, texts }
}

async function waitAssistantDone(page) {
  await page.waitForFunction(() => {
    const b = document.querySelectorAll('.message-row.assistant .message-bubble')
    const last = b[b.length - 1]
    return last && !last.classList.contains('streaming') && (last.textContent || '').trim().length > 20
  }, { timeout: 420000 })
}

async function main() {
  const report = { step: '06_rag_debug', timestamp: new Date().toISOString(), phases: {} }

  const created = await api('POST', '/users', {
    name: `Step06_API_${Date.now()}`,
    gender: 'female',
    birth_date: '19920618',
    height_cm: 165,
    weight_kg: 64,
    allergies: ['shellfish'],
    goal: 'lose_weight',
  })
  const apiUid = created.user.user_id
  report.api_user_id = apiUid
  report.phases.api_normal_sse = await parseSseStream(apiUid, NORMAL_MSG)
  report.phases.api_rag_sse = await parseSseStream(apiUid, RAG_MSG)

  const uiCreated = await api('POST', '/users', {
    name: `Step06_UI_${Date.now()}`,
    gender: 'female',
    birth_date: '19920618',
    height_cm: 165,
    weight_kg: 64,
    allergies: ['shellfish'],
    goal: 'lose_weight',
  })
  const uid = uiCreated.user.user_id
  report.user_id = uid

  const browser = await chromium.launch({ headless: true, channel: 'chrome' })
  const page = await browser.newPage({ viewport: { width: 1440, height: 900 } })
  page.setDefaultTimeout(420000)

  await page.goto(APP, { waitUntil: 'networkidle', timeout: 60000 })
  await page.waitForSelector('.hero-card h1', { timeout: 20000 })
  await page.reload({ waitUntil: 'networkidle' })
  await page.getByLabel('Open sidebar').click()
  await page.locator('.sidebar-body select.input').selectOption(String(uid))
  await page.getByLabel('Close sidebar').click()
  await page.waitForTimeout(500)
  await page.locator('.chat-section').scrollIntoViewIfNeeded()

  await page.locator('.chat-input').fill(NORMAL_MSG)
  await page.locator('.send-button').click()
  await waitAssistantDone(page)

  await page.locator('.chat-input').fill(RAG_MSG)
  await page.locator('.send-button').click()
  await waitAssistantDone(page)

  report.phases.ui_rag_immediate = await chipState(page)
  await page.waitForTimeout(1500)
  report.phases.ui_rag_after_refresh = await chipState(page)

  const chatFromApi = await api('GET', `/users/${uid}/chat`)
  const lastAssistant = [...(chatFromApi.messages || [])].reverse().find((m) => m.role === 'assistant')
  report.phases.api_chat_history_last_assistant = {
    has_sources_field: lastAssistant ? 'sources' in lastAssistant : false,
    keys: lastAssistant ? Object.keys(lastAssistant) : [],
  }

  report.diagnosis = {
    sse_meta_ok: report.phases.api_rag_sse.meta_sources.length > 0,
    sse_done_ok: report.phases.api_rag_sse.done_sources.length > 0,
    chips_before_refresh: report.phases.ui_rag_immediate.count,
    chips_after_refresh: report.phases.ui_rag_after_refresh.count,
    chat_api_persists_sources: report.phases.api_chat_history_last_assistant.has_sources_field,
    likely_loss_point:
      report.phases.api_rag_sse.done_sources.length > 0 &&
      report.phases.ui_rag_immediate.count > 0 &&
      report.phases.ui_rag_after_refresh.count === 0
        ? 'refreshUserData overwrites chatHistory (GET /chat has no sources)'
        : report.phases.api_rag_sse.done_sources.length > 0 &&
            report.phases.ui_rag_immediate.count === 0
          ? 'React streamChat/onDone path (before refresh)'
          : 'SSE or backend',
  }

  report.pass = report.phases.ui_rag_after_refresh.count > 0

  const out = path.join(ROOT, 'backend/eval/results/entry_021_step06_rag_debug.json')
  fs.writeFileSync(out, JSON.stringify(report, null, 2))
  console.log(JSON.stringify(report, null, 2))
  console.log('WROTE', out)

  await browser.close()
  process.exit(report.pass ? 0 : 1)
}

main().catch((e) => {
  console.error(e)
  process.exit(2)
})
