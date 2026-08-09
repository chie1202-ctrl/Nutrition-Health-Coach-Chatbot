#!/usr/bin/env node
/**
 * Entry 038 — Food-choice comparison UI validation.
 * Requires backend :8000 + frontend :5173.
 * Run: ./scripts/run_entry_038_food_choice_ui.sh
 * Or start manually: ./start.sh  (then node scripts/entry_038_food_choice_ui.mjs)
 */
import { chromium } from '../frontend/node_modules/playwright/index.mjs'
import fs from 'fs'
import path from 'path'
import { fileURLToPath } from 'url'

const __dirname = path.dirname(fileURLToPath(import.meta.url))
const ROOT = path.resolve(__dirname, '..')
const OUT_DIR = path.join(ROOT, 'backend/eval/results/entry_038_food_choice_ui')
const API = 'http://127.0.0.1:8000'
const APP = 'http://127.0.0.1:5173'
const VIEWPORT = { width: 1440, height: 900 }

const FC01_PROMPT =
  "I'm choosing between pizza and Chinese vegetable stir-fry for dinner — which fits my goal better?"
const NEGATIVE_PROMPT = 'What is a healthy breakfast?'

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

async function shot(page, name) {
  const file = path.join(OUT_DIR, name)
  await page.screenshot({ path: file, fullPage: false })
  return file
}

async function openSidebar(page) {
  if (!(await page.locator('.sidebar').isVisible())) {
    await page.getByLabel('Open sidebar').click()
    await page.waitForSelector('.sidebar-body')
  }
}

async function waitNoModal(page) {
  const backdrop = page.locator('.modal-backdrop')
  if (await backdrop.count()) {
    await backdrop.first().waitFor({ state: 'detached', timeout: 20000 })
  }
  await page.waitForTimeout(300)
}

async function waitChatReady(page, timeoutMs = 120000) {
  const healthFromBrowser = await page.evaluate(async () => {
    try {
      const response = await fetch('http://127.0.0.1:8000/health')
      if (!response.ok) return false
      const data = await response.json()
      return Boolean(data.ollama_reachable)
    } catch {
      return false
    }
  })
  if (!healthFromBrowser) {
    throw new Error('Browser cannot reach backend /health with ollama_reachable=true')
  }

  const uiReady = await page
    .locator('.chat-panel-header-text p')
    .filter({ hasText: 'Local AI active' })
    .count()
  if (!uiReady) {
    await page.reload({ waitUntil: 'networkidle', timeout: timeoutMs })
    await page.waitForResponse((response) => response.url().includes('/health') && response.ok(), {
      timeout: timeoutMs,
    })
  }

  await page.waitForFunction(
    () => document.querySelector('.chat-panel-header-text p')?.textContent?.includes('Local AI active'),
    { timeout: timeoutMs }
  )
}

async function sendChatAndWait(page, message, timeoutMs = 180000) {
  await waitNoModal(page)
  await waitChatReady(page)
  const streamPromise = page.waitForResponse(
    (response) => response.url().includes('/chat/stream') && response.request().method() === 'POST',
    { timeout: timeoutMs }
  )
  await page.locator('.chat-input').fill(message)
  await page.waitForFunction(
    () => {
      const button = document.querySelector('.send-button')
      return button && !button.disabled
    },
    { timeout: 30000 }
  )
  await page.locator('.send-button').click()
  const streamResponse = await streamPromise
  const streamBody = await streamResponse.text()
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
  return streamBody
}

function parseDonePayload(streamBody) {
  for (const block of streamBody.split('\n\n')) {
    if (!block.includes('event: done')) continue
    for (const line of block.split('\n')) {
      if (line.startsWith('data:')) return JSON.parse(line.slice(5))
    }
  }
  return null
}

async function sendChatAndWaitForFoodChoice(page, message, timeoutMs = 180000) {
  const streamBody = await sendChatAndWait(page, message, timeoutMs)
  const done = parseDonePayload(streamBody)
  const foodChoice = done?.food_choice
  if (!foodChoice || typeof foodChoice !== 'object' || !foodChoice.option_a) {
    const blocked = Boolean(done?.safety_blocked)
    throw new Error(
      `Done event missing food_choice comparison (safety_blocked=${blocked}). Reply excerpt: ${String(done?.reply || '').slice(0, 240)}`
    )
  }
  if (done.safety_blocked) {
    throw new Error(`Safety guardrail blocked food-choice response: ${String(done.reply || '').slice(0, 240)}`)
  }
  await page.waitForSelector('.message-row.assistant .food-choice-card', { timeout: 60000 })
}

function record(steps, key, pass, detail = '', evidence = null) {
  steps.push({ step: key, pass, detail, evidence })
  console.log(`${pass ? 'PASS' : 'FAIL'} ${key}: ${detail}`)
}

async function main() {
  fs.mkdirSync(OUT_DIR, { recursive: true })
  const steps = []

  const health = await api('GET', '/health')
  const envOk = health.ollama_reachable === true && health.rag_ready === true
  record(
    steps,
    '01_stack_health',
    envOk,
    `ollama=${health.ollama_reachable} rag=${health.rag_ready} model=${health.ollama_model}`,
    health
  )

  const browser = await chromium.launch({ headless: true, channel: 'chrome' })
  const page = await browser.newPage({ viewport: VIEWPORT })
  page.setDefaultTimeout(180000)

  await page.goto(APP, { waitUntil: 'networkidle', timeout: 60000 })
  await page.waitForResponse((response) => response.url().includes('/health') && response.ok(), {
    timeout: 60000,
  }).catch(() => {})
  await page.waitForSelector('.hero-card h1', { timeout: 20000 })
  record(steps, '02_react_ui_open', true, APP)
  await shot(page, '02_home.png')

  const demoName = `FC038_${Date.now()}`
  await openSidebar(page)
  await page.getByRole('button', { name: 'Add New User' }).click()
  await page.waitForSelector('.modal-form')
  await page.getByPlaceholder('Name').fill(demoName)
  await page.getByPlaceholder('Birth date (YYYYMMDD)').fill('19620101')
  await page.getByPlaceholder('Height (cm)').fill('165')
  await page.getByPlaceholder('Latest / initial weight (kg)').fill('72')
  await page.getByPlaceholder('Goal (e.g. lose_weight)').fill('lose_weight')
  await page.locator('#profile-allergies').getByRole('button', { name: 'Shellfish' }).click()
  await page.locator('.modal-card form.modal-form').evaluate((form) => form.requestSubmit())
  await waitNoModal(page)
  await page.waitForFunction(
    (name) => document.querySelector('.hero-card h1')?.textContent?.includes(name),
    demoName,
    { timeout: 15000 }
  )
  record(steps, '03_create_demo_user', true, demoName)
  const demoUserId = await page.locator('.sidebar-body select.input').inputValue()
  await shot(page, '03_user_created.png')

  await page.locator('.chat-section').scrollIntoViewIfNeeded()
  for (let attempt = 1; attempt <= 2; attempt += 1) {
    try {
      await sendChatAndWaitForFoodChoice(page, FC01_PROMPT)
      break
    } catch (err) {
      if (attempt === 2) throw err
      console.log(`WARN food-choice attempt ${attempt} failed, retrying: ${err.message}`)
      await page.waitForTimeout(1500)
    }
  }

  const lastAssistant = page.locator('.message-row.assistant').last()
  const cardVisible = await lastAssistant.locator('.food-choice-card').isVisible()
  record(steps, '04_food_choice_card_visible', cardVisible, cardVisible ? 'FoodChoiceCard rendered' : 'card missing')
  await shot(page, '04_food_choice_card.png')

  const tableRows = await lastAssistant.locator('.food-choice-table tbody tr').count()
  const tableOk = tableRows === 4
  record(steps, '05_comparison_table_rows', tableOk, `rows=${tableRows}`)

  const optionA = ((await lastAssistant.locator('.food-choice-option-a strong').textContent({ timeout: 5000 })) || '').trim()
  const optionB = ((await lastAssistant.locator('.food-choice-option-b strong').textContent({ timeout: 5000 })) || '').trim()
  const labelsOk = optionA.length > 0 && optionB.length > 0
  record(steps, '06_option_labels', labelsOk, `A="${optionA}" B="${optionB}"`)

  const sourceTexts = await lastAssistant.locator('.source-chip').allTextContents()
  const ragOk = sourceTexts.length >= 1
  record(
    steps,
    '07_rag_source_chips',
    ragOk,
    `chips=${sourceTexts.length} sources=${sourceTexts.join('; ')}`,
    { sources: sourceTexts }
  )

  await page.reload({ waitUntil: 'networkidle' })
  await page.waitForResponse((response) => response.url().includes('/health') && response.ok(), {
    timeout: 60000,
  }).catch(() => {})
  await openSidebar(page)
  await page.locator('.sidebar-body select.input').selectOption(demoUserId)
  await page.waitForFunction(
    (name) => document.querySelector('.hero-card h1')?.textContent?.includes(name),
    demoName,
    { timeout: 15000 }
  )
  await page.waitForSelector('.food-choice-card', { timeout: 30000 })
  const reloadCard = (await page.locator('.food-choice-card').count()) > 0
  record(steps, '08_reload_persists_card', reloadCard, reloadCard ? 'card after reload' : 'card lost')
  await shot(page, '08_after_reload.png')

  await page.locator('.chat-section').scrollIntoViewIfNeeded()
  const cardsBefore = await page.locator('.food-choice-card').count()
  await sendChatAndWait(page, NEGATIVE_PROMPT, 120000)
  const lastRow = page.locator('.message-row.assistant').last()
  const negativeHasCard = await lastRow.locator('.food-choice-card').isVisible()
  const cardsAfter = await page.locator('.food-choice-card').count()
  const negativeOk = !negativeHasCard && cardsAfter === cardsBefore
  record(
    steps,
    '09_negative_no_card',
    negativeOk,
    `negative_card=${negativeHasCard} total_cards=${cardsAfter}`
  )
  await shot(page, '09_negative_prompt.png')

  await browser.close()

  const allPass = steps.every((s) => s.pass)
  const report = {
    entry: '038',
    title: 'Food-choice comparison UI validation',
    timestamp: new Date().toISOString(),
    viewport: VIEWPORT,
    health_at_start: health,
    demo_user: demoName,
    prompt_fc01: FC01_PROMPT,
    prompt_negative: NEGATIVE_PROMPT,
    steps,
    acceptance: { entry_038_pass: allPass },
    screenshots_dir: OUT_DIR,
  }

  const outJson = path.join(ROOT, 'backend/eval/results/entry_038_food_choice_ui.json')
  fs.writeFileSync(outJson, JSON.stringify(report, null, 2))
  console.log('\nWROTE', outJson)
  console.log('Entry 038 UI', allPass ? 'PASS' : 'FAIL')
  process.exit(allPass ? 0 : 1)
}

main().catch((err) => {
  console.error(err)
  process.exit(1)
})
