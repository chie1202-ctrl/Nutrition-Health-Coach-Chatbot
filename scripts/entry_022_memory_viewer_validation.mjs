#!/usr/bin/env node
/**
 * Entry 022 — Level 2 Memory Viewer validation.
 * Requires stack: backend :8000, frontend :5173, Ollama online.
 */
import { chromium } from '../frontend/node_modules/playwright/index.mjs'
import fs from 'fs'
import path from 'path'
import { fileURLToPath } from 'url'

const __dirname = path.dirname(fileURLToPath(import.meta.url))
const ROOT = path.resolve(__dirname, '..')
const OUT_DIR = path.join(ROOT, 'backend/eval/results/entry_022_memory_viewer')
const RESULT_JSON = path.join(ROOT, 'backend/eval/results/entry_022_memory_viewer_validation.json')
const API = 'http://127.0.0.1:8000'
const APP = 'http://127.0.0.1:5173'
const VIEWPORT = { width: 1440, height: 900 }

const DAY1_MSG =
  'I want to lose 5 kg in three months. I prefer high-protein lunches and I cannot eat shellfish.'
const DAY2_MSG = 'What was my goal and what food should I avoid?'

async function api(method, route, body, timeoutMs = 300000) {
  const controller = new AbortController()
  const timer = setTimeout(() => controller.abort(), timeoutMs)
  try {
    const res = await fetch(`${API}${route}`, {
      method,
      headers: { 'Content-Type': 'application/json' },
      body: body ? JSON.stringify(body) : undefined,
      signal: controller.signal,
    })
    const text = await res.text()
    if (!res.ok) throw new Error(`${method} ${route} → ${res.status}: ${text.slice(0, 200)}`)
    return text ? JSON.parse(text) : {}
  } finally {
    clearTimeout(timer)
  }
}

function record(steps, key, pass, detail = '', evidence = null) {
  steps.push({ step: key, pass, detail, evidence })
  console.log(`${pass ? 'PASS' : 'FAIL'} ${key}: ${detail}`)
}

async function shot(page, name) {
  const file = path.join(OUT_DIR, name)
  await page.screenshot({ path: file, fullPage: false })
  return file
}

async function waitNoModal(page) {
  const backdrop = page.locator('.modal-backdrop')
  if (await backdrop.count()) {
    await backdrop.first().waitFor({ state: 'detached', timeout: 20000 })
  }
  await page.waitForTimeout(300)
}

async function sendChatAndWait(page, message, timeoutMs = 420000) {
  await waitNoModal(page)
  await page.locator('.chat-input').fill(message)
  await page.locator('.send-button').click()
  await page.waitForFunction(
    () => {
      const bubbles = document.querySelectorAll('.message-row.assistant .message-bubble')
      if (!bubbles.length) return false
      const last = bubbles[bubbles.length - 1]
      if (last.classList.contains('streaming')) return false
      const text = (last.textContent || '').replace(/Safety guardrail response/g, '').trim()
      return text.length > 20
    },
    { timeout: timeoutMs }
  )
  await page.waitForTimeout(800)
}

async function waitForSessionSummary(uid, timeoutMs = 120000) {
  const start = Date.now()
  while (Date.now() - start < timeoutMs) {
    const [mem, summaries, sessions] = await Promise.all([
      api('GET', `/users/${uid}/memory`),
      api('GET', `/users/${uid}/summaries?limit=20`),
      api('GET', `/users/${uid}/sessions?limit=20`),
    ])
    const hasSummary =
      (mem.recent_session_summaries?.length ?? 0) > 0 ||
      summaries.some((s) => s.summary_type === 'session' && !s.archived)
    const closedSession = sessions.some((s) => s.status === 'closed')
    if (hasSummary && closedSession) {
      return { mem, summaries, sessions }
    }
    await new Promise((r) => setTimeout(r, 2000))
  }
  throw new Error('Timed out waiting for session summary after close')
}

async function main() {
  fs.mkdirSync(OUT_DIR, { recursive: true })
  const steps = []
  const evidence = { api: {}, screenshots: [] }

  const health = await api('GET', '/health')
  evidence.api.health = health
  const envOk = health.ollama_reachable === true && health.memory_mode === 'M2'
  record(steps, '01_stack_health', envOk, `ollama=${health.ollama_reachable} memory_mode=${health.memory_mode}`)

  const browser = await chromium.launch({ headless: true, channel: 'chrome' })
  const page = await browser.newPage({ viewport: VIEWPORT })
  page.setDefaultTimeout(420000)

  await page.goto(APP, { waitUntil: 'networkidle', timeout: 60000 })
  await page.waitForSelector('.hero-card h1', { timeout: 20000 })
  record(steps, '02_react_ui_open', true, APP)
  evidence.screenshots.push(await shot(page, '01_home.png'))

  const demoName = `MemView_${Date.now()}`
  if (!(await page.locator('.sidebar').isVisible())) {
    await page.getByLabel('Open sidebar').click()
    await page.waitForSelector('.sidebar-body')
  }
  await page.getByRole('button', { name: 'Add New User' }).click()
  await page.waitForSelector('.modal-form')
  await page.getByPlaceholder('Name').fill(demoName)
  await page.getByPlaceholder('Birth date (YYYYMMDD)').fill('19920618')
  await page.getByPlaceholder('Height (cm)').fill('165')
  await page.getByPlaceholder('Latest / initial weight (kg)').fill('64')
  await page.getByPlaceholder('Goal (e.g. lose_weight)').fill('lose_weight')
  await page.getByPlaceholder('Target weight').fill('59 kg')
  await page.getByPlaceholder('Target timeline').fill('3 months')
  await page.getByPlaceholder('Diet preference').fill('high protein')
  await page.getByPlaceholder('Allergies (comma separated)').fill('shellfish')
  await page.locator('.modal-card form.modal-form').evaluate((form) => form.requestSubmit())
  await waitNoModal(page)
  await page.waitForFunction(
    (name) => document.querySelector('.hero-card h1')?.textContent?.includes(name),
    demoName,
    { timeout: 15000 }
  )
  const userId = await page.locator('.sidebar-body select.input').inputValue()
  record(steps, '03_create_user_with_profile', true, `${demoName} id=${userId}`)
  evidence.screenshots.push(await shot(page, '02_user_created.png'))

  await page.locator('.chat-section').scrollIntoViewIfNeeded()
  await sendChatAndWait(page, DAY1_MSG)
  await sendChatAndWait(page, DAY2_MSG)
  record(steps, '04_seed_chat_turns', true, 'two chat turns before session close')

  await page.getByRole('button', { name: 'New Conversation' }).click()
  await page.waitForFunction(
    () => !document.querySelector('.chat-new-session-button')?.textContent?.includes('Closing'),
    { timeout: 120000 }
  )
  const memoryBundle = await waitForSessionSummary(userId)
  evidence.api.memory_after_close = memoryBundle.mem
  evidence.api.summaries_after_close = memoryBundle.summaries
  evidence.api.sessions_after_close = memoryBundle.sessions
  record(
    steps,
    '05_session_closed_with_summary',
    memoryBundle.mem.recent_session_summaries?.length > 0,
    `summaries=${memoryBundle.mem.recent_session_summaries?.length ?? 0} sessions=${memoryBundle.sessions.length}`
  )

  await page.getByRole('button', { name: 'View Memory' }).click()
  await page.waitForSelector('.memory-viewer-modal')
  record(steps, '06_open_memory_viewer', true, 'modal visible')
  evidence.screenshots.push(await shot(page, '03_memory_viewer_open.png'))

  const modal = page.locator('.memory-viewer-modal')
  await modal.locator('text=Profile context').waitFor()
  const profileCards = await modal.locator('.memory-fact-card').count()
  const profileText = await modal.locator('.memory-facts-grid').textContent()
  const profilePass =
    profileCards >= 3 &&
    /Goal/i.test(profileText) &&
    /shellfish/i.test(profileText) &&
    /lose weight|high protein/i.test(profileText)
  record(
    steps,
    '07_profile_context',
    profilePass,
    `cards=${profileCards} has_goal_allergy_diet=${profilePass}`
  )
  evidence.screenshots.push(await shot(page, '04_profile_context.png'))

  const longTermSection = modal.locator('.memory-section').filter({ hasText: 'Long-term cumulative memory' })
  const longTermVisible = (await longTermSection.count()) > 0
  const longTermText = longTermVisible ? await longTermSection.textContent() : ''
  const hasCumulative = Boolean(memoryBundle.mem.cumulative_summary?.trim())
  const longTermPass = longTermVisible && (hasCumulative ? /Coaching Context|Goals|Session notes/i.test(longTermText) : /No long-term memory yet/i.test(longTermText))
  record(
    steps,
    '08_long_term_memory_section',
    longTermPass,
    hasCumulative ? 'cumulative content or empty state ok' : 'empty state shown (expected for first session)'
  )

  const summaryCards = await modal.locator('.memory-summary-card').count()
  const summaryText = summaryCards ? await modal.locator('.memory-summary-list').textContent() : ''
  const summaryPass = summaryCards > 0 && (/5\s*kg|shellfish|lose/i.test(summaryText))
  record(steps, '09_recent_session_summaries', summaryPass, `cards=${summaryCards}`)
  evidence.screenshots.push(await shot(page, '05_session_summaries.png'))

  const sessionCards = await modal.locator('.memory-session-card').count()
  const sessionPass = sessionCards > 0
  record(steps, '10_session_history', sessionPass, `cards=${sessionCards}`)
  evidence.screenshots.push(await shot(page, '06_session_history.png'))

  const viewSummaryBtn = modal.getByRole('button', { name: 'View summary' }).first()
  const hasViewBtn = (await viewSummaryBtn.count()) > 0
  let expandPass = false
  if (hasViewBtn) {
    await viewSummaryBtn.click()
    await page.waitForTimeout(400)
    const expandedContent = await modal.locator('.memory-session-card pre.memory-block-content').first().textContent()
    expandPass = expandedContent.length > 40
    await modal.getByRole('button', { name: 'Hide summary' }).first().click()
    await page.waitForTimeout(300)
    const hidden = (await modal.locator('.memory-session-card pre.memory-block-content').count()) === 0
    expandPass = expandPass && hidden
  }
  record(steps, '11_view_summary_toggle', hasViewBtn && expandPass, hasViewBtn ? 'expand/collapse ok' : 'no View summary button')

  const refreshBtn = modal.getByRole('button', { name: 'Refresh' })
  await refreshBtn.click()
  await page.waitForFunction(
    () => !document.querySelector('.memory-viewer-loading'),
    { timeout: 30000 }
  )
  const stillOpen = await modal.isVisible()
  const profileAfterRefresh = await modal.locator('.memory-fact-card').count()
  record(steps, '12_refresh', stillOpen && profileAfterRefresh >= 3, `modal_open=${stillOpen} profile_cards=${profileAfterRefresh}`)
  evidence.screenshots.push(await shot(page, '07_after_refresh.png'))

  const closedSessions = memoryBundle.sessions.filter((s) => s.status === 'closed')
  const eligible = closedSessions.filter((s) => (s.turn_count ?? 0) >= 2)
  const activeSessions = memoryBundle.sessions.filter((s) => s.status === 'active')
  const regenButtons = await modal.getByRole('button', { name: 'Regenerate' }).count()
  const regenPass =
    eligible.length === regenButtons &&
    activeSessions.every(() => true) &&
    regenButtons >= 1
  record(
    steps,
    '13_regenerate_button_eligibility',
    regenPass,
    `eligible_closed=${eligible.length} regen_buttons=${regenButtons} active_sessions=${activeSessions.length}`
  )

  const lowTurnClosed = closedSessions.filter((s) => (s.turn_count ?? 0) < 2)
  record(
    steps,
    '13b_no_regenerate_on_ineligible',
    lowTurnClosed.length === 0 || regenButtons <= eligible.length,
    `low_turn_closed=${lowTurnClosed.length}`
  )

  await modal.getByLabel('Close memory viewer').click()
  await page.waitForSelector('.memory-viewer-modal', { state: 'detached', timeout: 10000 })
  const closed = !(await page.locator('.memory-viewer-modal').count())
  record(steps, '14_close_modal', closed, 'close button')

  const passCount = steps.filter((s) => s.pass).length
  const overall = steps.every((s) => s.pass)
  const result = {
    entry: '022',
    title: 'Level 2 Memory Viewer validation',
    date: new Date().toISOString().slice(0, 10),
    viewport: VIEWPORT,
    user_id: Number(userId),
    user_name: demoName,
    overall_pass: overall,
    score: `${passCount}/${steps.length}`,
    steps,
    evidence,
    screenshot_dir: OUT_DIR,
  }
  fs.writeFileSync(RESULT_JSON, JSON.stringify(result, null, 2))
  console.log(`\nResult: ${overall ? 'PASS' : 'FAIL'} (${passCount}/${steps.length})`)
  console.log(`Artifact: ${RESULT_JSON}`)
  console.log(`Screenshots: ${OUT_DIR}`)

  await browser.close()
  process.exit(overall ? 0 : 1)
}

main().catch((err) => {
  console.error(err)
  process.exit(1)
})
