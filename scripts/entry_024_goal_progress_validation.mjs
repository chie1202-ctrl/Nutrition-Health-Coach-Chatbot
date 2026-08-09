#!/usr/bin/env node
/**
 * Entry 024 — Level 2 Goal Progress validation.
 */
import { chromium } from '../frontend/node_modules/playwright/index.mjs'
import fs from 'fs'
import path from 'path'
import { fileURLToPath } from 'url'

const __dirname = path.dirname(fileURLToPath(import.meta.url))
const ROOT = path.resolve(__dirname, '..')
const OUT_DIR = path.join(ROOT, 'backend/eval/results/entry_024_goal_progress')
const RESULT_JSON = path.join(ROOT, 'backend/eval/results/entry_024_goal_progress_validation.json')
const API = 'http://127.0.0.1:8000'
const APP = 'http://127.0.0.1:5173'

async function api(method, route, body) {
  const res = await fetch(`${API}${route}`, {
    method,
    headers: { 'Content-Type': 'application/json' },
    body: body ? JSON.stringify(body) : undefined,
  })
  const text = await res.text()
  if (!res.ok) throw new Error(`${method} ${route} → ${res.status}: ${text.slice(0, 200)}`)
  return text ? JSON.parse(text) : {}
}

function record(steps, key, pass, detail = '') {
  steps.push({ step: key, pass, detail })
  console.log(`${pass ? 'PASS' : 'FAIL'} ${key}: ${detail}`)
}

function daysAgo(n) {
  const d = new Date()
  d.setDate(d.getDate() - n)
  const pad = (v) => String(v).padStart(2, '0')
  return `${d.getFullYear()}-${pad(d.getMonth() + 1)}-${pad(d.getDate())} 10:00:00`
}

async function prepareUser() {
  const users = await api('GET', '/users')
  let user = users.find((item) => item.name.startsWith('MemView_')) || users.find((item) => item.target_weight) || users[0]
  if (!user) throw new Error('No users available for goal progress validation')

  const profile = await api('GET', `/users/${user.user_id}`)
  await api('PUT', `/users/${user.user_id}`, {
    name: user.name,
    gender: user.gender,
    birth_date: user.birth_date,
    height_cm: user.height_cm,
    goal: 'lose_weight',
    target_weight: '58 kg',
    target_timeline: '3 months',
    activity_level: user.activity_level || '',
    diet_preference: user.diet_preference || 'high protein',
    budget_level: user.budget_level || '',
    medical_conditions: user.medical_conditions || [],
    allergies: user.allergies || [],
    food_dislikes: user.food_dislikes || [],
    self_description: user.self_description || '',
    coach_notes: user.coach_notes || '',
  })

  const weights = [
    { weight_kg: 65, recorded_at: daysAgo(14) },
    { weight_kg: 64.2, recorded_at: daysAgo(7) },
    { weight_kg: 63.5, recorded_at: daysAgo(0) },
  ]
  for (const entry of weights) {
    await api('POST', `/users/${user.user_id}/weight`, {
      user_id: user.user_id,
      weight_kg: entry.weight_kg,
      recorded_at: entry.recorded_at,
      note: 'Goal progress validation seed',
    })
  }

  const bundle = await api('GET', `/users/${user.user_id}`)
  return { user: bundle.user, metrics: bundle.metrics }
}

async function validateViewport(page, viewportName, userId, expected) {
  const steps = []
  await page.goto(APP, { waitUntil: 'networkidle' })
  if (!(await page.locator('.sidebar').isVisible())) {
    await page.getByLabel('Open sidebar').click()
  }
  await page.locator('.sidebar-body select.input').selectOption(String(userId))
  await page.waitForSelector('.goal-progress-section', { timeout: 15000 })
  await page.screenshot({ path: path.join(OUT_DIR, `${viewportName}_goal_progress.png`), fullPage: false })

  const cardText = await page.locator('.goal-progress-card').textContent()
  record(steps, `${viewportName}_card_visible`, /Goal Progress/i.test(cardText || ''), 'section rendered')
  record(steps, `${viewportName}_current_weight`, cardText.includes('63.5'), `text=${cardText?.slice(0, 120)}`)
  record(steps, `${viewportName}_target_weight`, /58\s*kg/i.test(cardText || ''), 'target shown')
  record(steps, `${viewportName}_remaining`, /5\.5\s*kg/i.test(cardText || ''), 'remaining to target')
  record(steps, `${viewportName}_percent`, cardText.includes(`${expected.progressPercent}%`), `expected ${expected.progressPercent}%`)
  record(steps, `${viewportName}_trend`, /weigh-in|logged weigh-in|Trending/i.test(cardText || ''), 'trend message present')

  return steps
}

async function main() {
  fs.mkdirSync(OUT_DIR, { recursive: true })
  const steps = []

  await api('GET', '/health')
  record(steps, '01_stack_health', true, 'backend ok')

  const prepared = await prepareUser()
  const userId = prepared.user.user_id
  const series = prepared.metrics.series || []
  record(steps, '02_seed_user_history', series.length >= 3, `user_id=${userId} series=${series.length}`)

  const start = Number(series[0]?.weight_kg)
  const current = Number(prepared.metrics.weight_kg)
  const target = 58
  const progressPercent = Math.round(((start - current) / (start - target)) * 100)
  const expected = { progressPercent, remaining: current - target }

  for (const viewport of [
    { name: 'desktop', width: 1440, height: 900 },
    { name: 'mobile', width: 390, height: 844 },
  ]) {
    const browser = await chromium.launch({ headless: true, channel: 'chrome' })
    const page = await browser.newPage({ viewport: { width: viewport.width, height: viewport.height } })
    const viewSteps = await validateViewport(page, viewport.name, userId, expected)
    steps.push(...viewSteps)
    await browser.close()
  }

  const overall = steps.every((s) => s.pass)
  const result = {
    entry: '024',
    title: 'Level 2 Goal Progress validation',
    date: new Date().toISOString().slice(0, 10),
    user_id: userId,
    expected,
    overall_pass: overall,
    score: `${steps.filter((s) => s.pass).length}/${steps.length}`,
    steps,
    screenshot_dir: OUT_DIR,
  }
  fs.writeFileSync(RESULT_JSON, JSON.stringify(result, null, 2))
  console.log(`\nResult: ${overall ? 'PASS' : 'FAIL'} (${result.score})`)
  console.log(`Artifact: ${RESULT_JSON}`)
  process.exit(overall ? 0 : 1)
}

main().catch((err) => {
  console.error(err)
  process.exit(1)
})
