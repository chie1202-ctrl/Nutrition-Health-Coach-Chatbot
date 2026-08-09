#!/usr/bin/env node
/**
 * Entry 021 — Steps 07c and 08 only (post-blocker fixes).
 * Does not modify entry_021_b1_desktop_validation.mjs.
 */
import { chromium } from '../frontend/node_modules/playwright/index.mjs'
import fs from 'fs'
import path from 'path'
import { fileURLToPath } from 'url'

const __dirname = path.dirname(fileURLToPath(import.meta.url))
const ROOT = path.resolve(__dirname, '..')
const API = 'http://127.0.0.1:8000'
const APP = 'http://127.0.0.1:5173'
const OUT = path.join(ROOT, 'backend/eval/results/entry_021_step07c_08_check.json')

const DAY1_MSG =
  'I want to lose 5 kg in three months. I prefer high-protein lunches and I cannot eat shellfish.'
const DAY2_MSG = 'What was my goal and what food should I avoid?'

async function api(method, route, body, timeoutMs = 420000) {
  const controller = new AbortController()
  const timer = setTimeout(() => controller.abort(), timeoutMs)
  const res = await fetch(`${API}${route}`, {
    method,
    headers: { 'Content-Type': 'application/json' },
    signal: controller.signal,
    body: body ? JSON.stringify(body) : undefined,
  })
  clearTimeout(timer)
  if (!res.ok) throw new Error(`${method} ${route} → ${res.status}`)
  return res.json()
}

async function streamChat(uid, message, forceNew = false) {
  const res = await fetch(`${API}/chat/stream`, {
    method: 'POST',
    headers: { 'Content-Type': 'application/json' },
    body: JSON.stringify({ user_id: uid, message, force_new_session: forceNew }),
  })
  let reply = ''
  const text = await res.text()
  for (const block of text.split('\n\n')) {
    if (!block.trim()) continue
    let ev = 'message'
    let data = null
    for (const line of block.split('\n')) {
      if (line.startsWith('event:')) ev = line.slice(6).trim()
      if (line.startsWith('data:')) data = JSON.parse(line.slice(5))
    }
    if (ev === 'done') reply = data?.reply || reply
  }
  return reply
}

async function waitAssistantDone(page) {
  await page.waitForFunction(() => {
    const b = document.querySelectorAll('.message-row.assistant .message-bubble')
    const last = b[b.length - 1]
    return last && !last.classList.contains('streaming') && (last.textContent || '').trim().length > 20
  }, { timeout: 420000 })
}

async function sendChatAndWait(page, message) {
  await page.locator('.chat-input').fill(message)
  await page.locator('.send-button').click()
  await waitAssistantDone(page)
  const last = page.locator('.message-row.assistant').last()
  return (await last.locator('.message-bubble').textContent()) || ''
}

async function main() {
  const report = { timestamp: new Date().toISOString(), steps: {} }

  const health = await api('GET', '/health')
  report.health = health

  const created = await api('POST', '/users', {
    name: `Step07c08_${Date.now()}`,
    gender: 'female',
    birth_date: '19920618',
    height_cm: 165,
    weight_kg: 64,
    allergies: ['shellfish'],
    goal: 'lose_weight',
  })
  const uid = created.user.user_id
  report.user_id = uid

  // --- 07c via API (same memory path as UI) ---
  await streamChat(uid, DAY1_MSG)
  await api('POST', `/users/${uid}/sessions/close`)
  for (let i = 0; i < 60; i++) {
    const mem = await api('GET', `/users/${uid}/memory`)
    if ((mem.recent_session_summaries || []).length > 0) break
    await new Promise((r) => setTimeout(r, 2000))
  }
  const day2Reply = await streamChat(uid, DAY2_MSG, true)
  const low = day2Reply.toLowerCase()
  const pass07c = (/lose\s*5|5\s*kg/i.test(day2Reply) || /5\s*kg/i.test(day2Reply)) && low.includes('shellfish')
  report.steps['07c_cross_session_day2_recall'] = {
    pass: pass07c,
    has_lose_5kg: /lose\s*5|5\s*kg/i.test(day2Reply),
    has_shellfish: low.includes('shellfish'),
    reply_excerpt: day2Reply.slice(0, 320),
  }

  // --- 08 via UI (matches harness meal-plan button path) ---
  const browser = await chromium.launch({ headless: true, channel: 'chrome' })
  const page = await browser.newPage({ viewport: { width: 1440, height: 900 } })
  page.setDefaultTimeout(420000)

  let mealPlanApi = null
  page.on('response', async (response) => {
    if (response.url().includes('/meal-plan') && response.request().method() === 'POST' && response.ok()) {
      try {
        mealPlanApi = await response.json()
      } catch {
        // ignore
      }
    }
  })

  await page.goto(APP, { waitUntil: 'networkidle', timeout: 60000 })
  await page.getByLabel('Open sidebar').click()
  await page.locator('.sidebar-body select.input').selectOption(String(uid))
  await page.getByLabel('Close sidebar').click()
  await page.waitForTimeout(500)

  await page.locator('.plan-section').scrollIntoViewIfNeeded()
  await page.locator('.plan-generate-button').click()
  await page.waitForFunction(
    () => !document.querySelector('.plan-generate-button')?.textContent?.includes('Generating'),
    { timeout: 180000 }
  )
  await page.waitForTimeout(1000)

  const mealPlanNav = await page.locator('.meal-plan-nav').count()
  const mp = mealPlanApi || {}
  const validation = mp.validation || {}
  const pass08 =
    mp.llm_degraded === false &&
    validation.valid === true &&
    mealPlanNav >= 1 &&
    validation.day_count === 7 &&
    (validation.distinct_main_meals || 0) >= 5

  report.steps['08_live_meal_plan'] = {
    pass: pass08,
    llm_degraded: mp.llm_degraded,
    validation,
    meal_plan_nav: mealPlanNav,
    summary_excerpt: (mp.plan?.summary || '').slice(0, 160),
  }

  await browser.close()

  report.pass = pass07c && pass08
  fs.writeFileSync(OUT, JSON.stringify(report, null, 2))
  console.log(JSON.stringify(report, null, 2))
  console.log('WROTE', OUT)
  process.exit(report.pass ? 0 : 1)
}

main().catch((e) => {
  console.error(e)
  process.exit(2)
})
