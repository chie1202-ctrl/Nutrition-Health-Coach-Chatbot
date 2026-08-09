#!/usr/bin/env node
/**
 * Entry 020 — Mobile viewport validation (375×812, 390×844).
 * Requires backend :8000 and frontend :5173.
 */
import { chromium } from '../frontend/node_modules/playwright/index.mjs'
import fs from 'fs'
import path from 'path'
import { fileURLToPath } from 'url'

const __dirname = path.dirname(fileURLToPath(import.meta.url))
const ROOT = path.resolve(__dirname, '..')
const OUT_DIR = path.join(ROOT, 'backend/eval/results/entry_020_mobile')
const API = 'http://127.0.0.1:8000'
const APP = 'http://127.0.0.1:5173'

const VIEWPORTS = [
  { id: '375x812', width: 375, height: 812 },
  { id: '390x844', width: 390, height: 844 },
]

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
    if (!res.ok) throw new Error(`${method} ${route} → ${res.status}`)
    return res.json()
  } finally {
    clearTimeout(timer)
  }
}

async function ensureTestUser() {
  const health = await api('GET', '/health')
  const suffix = Date.now()
  const created = await api('POST', '/users', {
    name: `Mobile_E2E_${suffix}`,
    gender: 'female',
    birth_date: '1992-03-15',
    height_cm: 165,
    weight_kg: 62,
    allergies: ['shellfish'],
    goal: 'lose_weight',
  })
  const uid = created.user.user_id
  await api('POST', `/users/${uid}/weight`, {
    user_id: uid,
    weight_kg: 61.2,
    note: 'mobile validation',
  })
  return { health, uid, name: created.user.name }
}

async function seedMemory(uid) {
  await api('POST', '/chat', {
    user_id: uid,
    message: 'I walked 30 minutes today and prefer high-protein lunches.',
  })
  await api('POST', `/users/${uid}/sessions/close`, {})
}

async function seedMealPlan(uid) {
  return api('POST', `/users/${uid}/meal-plan`)
}

function checkOverflow(page) {
  return page.evaluate(() => {
    const doc = document.documentElement
    const body = document.body
    const maxW = Math.max(doc.scrollWidth, body?.scrollWidth || 0)
    const clientW = doc.clientWidth
    const offenders = []
    document.querySelectorAll('*').forEach((el) => {
      const r = el.getBoundingClientRect()
      if (r.right > clientW + 2 && r.width > 0) {
        const tag = `${el.tagName.toLowerCase()}${el.className ? '.' + String(el.className).split(' ')[0] : ''}`
        offenders.push({ tag, right: Math.round(r.right), clientW })
      }
    })
    offenders.sort((a, b) => b.right - a.right)
    return {
      scrollWidth: maxW,
      clientWidth: clientW,
      overflow: maxW > clientW + 1,
      topOffenders: offenders.slice(0, 8),
    }
  })
}

function rect(page, selector) {
  return page.locator(selector).first().boundingBox()
}

async function validateViewport(browser, viewport, ctx) {
  const { uid, name } = ctx
  const page = await browser.newPage({ viewport: { width: viewport.width, height: viewport.height } })
  const shots = path.join(OUT_DIR, viewport.id)
  fs.mkdirSync(shots, { recursive: true })
  const checks = {}
  const issues = []

  const record = (key, pass, detail = '') => {
    checks[key] = { pass, detail }
    if (!pass) issues.push({ check: key, detail })
  }

  await page.goto(APP, { waitUntil: 'networkidle', timeout: 30000 })
  await page.waitForSelector('.hero-card h1', { timeout: 15000 })

  // Mobile loads with sidebar open — close via header control (backdrop center sits under sidebar on narrow widths)
  const sidebarVisible = await page.locator('.sidebar').isVisible()
  record('sidebar_initial_state', true, sidebarVisible ? 'sidebar open on load (mobile overlay)' : 'sidebar closed')
  if (sidebarVisible) {
    await page.getByLabel('Close sidebar').click()
    await page.waitForTimeout(300)
  }
  await page.screenshot({ path: path.join(shots, '01_home_sidebar_closed.png'), fullPage: false })

  // 1. Sidebar navigation
  await page.getByLabel('Open sidebar').click()
  await page.waitForSelector('.sidebar-body select.input')
  record('sidebar_opens', await page.locator('.sidebar').isVisible())
  await page.screenshot({ path: path.join(shots, '02_sidebar_open.png'), fullPage: false })

  const selectBox = await rect(page, '.sidebar-body select.input')
  record(
    'user_select_visible',
    Boolean(selectBox && selectBox.width >= 44 && selectBox.height >= 36),
    selectBox ? `${Math.round(selectBox.width)}×${Math.round(selectBox.height)}` : 'missing',
  )

  // Select test user
  await page.locator('.sidebar-body select.input').selectOption(String(uid))
  await page.waitForTimeout(800)
  const heroText = await page.locator('.hero-card h1').textContent()
  record('user_selection_updates_view', heroText?.includes(name) ?? false, heroText || '')

  const chartCard = page.locator('.chart-card.chart-button').first()
  record('weight_chart_sidebar', await chartCard.isVisible())
  if (await chartCard.isVisible()) {
    await chartCard.click()
    await page.waitForSelector('.modal-card', { timeout: 5000 })
    await page.screenshot({ path: path.join(shots, '03_weight_chart_modal.png'), fullPage: false })
    const modalBox = await rect(page, '.modal-card')
    record(
      'weight_chart_modal_fits',
      Boolean(modalBox && modalBox.width <= viewport.width),
      modalBox ? `modal ${Math.round(modalBox.width)}px` : 'no modal',
    )
    await page.locator('.modal-close').first().click()
    await page.waitForSelector('.modal-card', { state: 'hidden', timeout: 5000 }).catch(() => {})
  } else {
    record('weight_chart_sidebar', false, 'no chart card')
    record('weight_chart_modal_fits', true, 'skipped — no chart data')
  }

  if (await page.locator('.sidebar').isVisible()) {
    await page.getByLabel('Close sidebar').click()
    await page.waitForTimeout(200)
  }

  // 6. Medical disclaimer
  const disclaimer = page.locator('.medical-disclaimer')
  record('medical_disclaimer_visible', await disclaimer.isVisible())
  const discBox = await disclaimer.boundingBox()
  record(
    'medical_disclaimer_readable',
    Boolean(discBox && discBox.width > 200 && discBox.height > 20),
    discBox ? `${Math.round(discBox.width)}×${Math.round(discBox.height)}` : 'missing',
  )

  // 7. Horizontal overflow (main view)
  const overflowMain = await checkOverflow(page)
  record('no_horizontal_overflow', !overflowMain.overflow, JSON.stringify(overflowMain.topOffenders.slice(0, 3)))

  // 4. Meal plan section
  await page.locator('.plan-section').scrollIntoViewIfNeeded()
  await page.screenshot({ path: path.join(shots, '04_meal_plan_section.png'), fullPage: false })
  const genBtn = page.locator('.plan-generate-button')
  const genBox = await genBtn.boundingBox()
  record(
    'meal_plan_generate_control',
    Boolean(genBox && genBox.width <= viewport.width && genBox.height >= 40),
    genBox ? `${Math.round(genBox.width)}×${Math.round(genBox.height)}` : 'missing',
  )
  const mealCards = await page.locator('.meal-card').count()
  record('meal_plan_content_visible', mealCards >= 1, `${mealCards} meal cards`)

  // 2. Chat interface
  await page.locator('.chat-section').scrollIntoViewIfNeeded()
  await page.screenshot({ path: path.join(shots, '05_chat_section.png'), fullPage: false })
  const chatInput = page.locator('.chat-input')
  const sendBtn = page.locator('.send-button')
  const inputBox = await chatInput.boundingBox()
  const sendBox = await sendBtn.boundingBox()
  record(
    'chat_input_visible',
    Boolean(inputBox && inputBox.width > 100),
    inputBox ? `${Math.round(inputBox.width)}×${Math.round(inputBox.height)}` : 'missing',
  )
  record(
    'chat_send_tap_target',
    Boolean(sendBox && sendBox.width >= 40 && sendBox.height >= 40),
    sendBox ? `${Math.round(sendBox.width)}×${Math.round(sendBox.height)}` : 'missing',
  )

  // 3. Cross-session memory UI
  const newSessionBtn = page.locator('.chat-new-session-button')
  const nsBox = await newSessionBtn.boundingBox()
  record(
    'new_conversation_control',
    Boolean(nsBox && nsBox.width <= viewport.width && nsBox.height >= 40),
    nsBox ? `${Math.round(nsBox.width)}×${Math.round(nsBox.height)}` : 'missing',
  )
  const memoryHint = await page.locator('.chat-heading-row p').textContent()
  record(
    'cross_session_memory_hint',
    memoryHint?.includes('Coach remembers') ?? false,
    memoryHint || '',
  )

  // Send button not obscured by fixed input shell
  const overlap = await page.evaluate(() => {
    const inputShell = document.querySelector('.chat-input-shell')
    const send = document.querySelector('.send-button')
    if (!inputShell || !send) return { ok: false, reason: 'missing elements' }
    const s = inputShell.getBoundingClientRect()
    const b = send.getBoundingClientRect()
    return {
      ok: b.bottom <= window.innerHeight + 1 && b.top >= s.top - 2,
      shellBottom: s.bottom,
      sendBottom: b.bottom,
      innerHeight: window.innerHeight,
    }
  })
  record('chat_controls_not_clipped', overlap.ok, JSON.stringify(overlap))

  await page.screenshot({ path: path.join(shots, '06_full_page.png'), fullPage: true })

  const allPass = issues.length === 0
  await page.close()
  return { viewport: viewport.id, checks, issues, pass: allPass }
}

async function main() {
  fs.mkdirSync(OUT_DIR, { recursive: true })
  const { health, uid, name } = await ensureTestUser()
  await seedMemory(uid)
  const meal = await seedMealPlan(uid)

  const browser = await chromium.launch({ headless: true, channel: 'chrome' })
  const results = []
  for (const vp of VIEWPORTS) {
    results.push(await validateViewport(browser, vp, { uid, name }))
  }
  await browser.close()

  const report = {
    entry: '020',
    timestamp: new Date().toISOString(),
    health_at_start: health,
    test_user_id: uid,
    meal_plan_llm_degraded: meal.llm_degraded,
    viewports: results,
    acceptance: {
      both_viewports_pass: results.every((r) => r.pass),
      entry_020_pass: results.every((r) => r.pass),
    },
    screenshots_dir: OUT_DIR,
  }

  const outJson = path.join(ROOT, 'backend/eval/results/entry_020_mobile_validation_20260619.json')
  fs.writeFileSync(outJson, JSON.stringify(report, null, 2))
  console.log(JSON.stringify(report, null, 2))
  process.exit(report.acceptance.entry_020_pass ? 0 : 1)
}

main().catch((err) => {
  console.error(err)
  process.exit(2)
})
