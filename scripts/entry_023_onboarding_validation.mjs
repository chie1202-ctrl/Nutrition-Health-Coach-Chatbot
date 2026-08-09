#!/usr/bin/env node
/**
 * Entry 023 — Level 2 Onboarding validation.
 * Uses ?onboarding=preview to exercise the flow when users already exist.
 */
import { chromium } from '../frontend/node_modules/playwright/index.mjs'
import fs from 'fs'
import path from 'path'
import { fileURLToPath } from 'url'

const __dirname = path.dirname(fileURLToPath(import.meta.url))
const ROOT = path.resolve(__dirname, '..')
const OUT_DIR = path.join(ROOT, 'backend/eval/results/entry_023_onboarding')
const RESULT_JSON = path.join(ROOT, 'backend/eval/results/entry_023_onboarding_validation.json')
const API = 'http://127.0.0.1:8000'
const APP = 'http://127.0.0.1:5173/?onboarding=preview'

function record(steps, key, pass, detail = '') {
  steps.push({ step: key, pass, detail })
  console.log(`${pass ? 'PASS' : 'FAIL'} ${key}: ${detail}`)
}

async function main() {
  fs.mkdirSync(OUT_DIR, { recursive: true })
  const steps = []
  const demoName = `Onboard_${Date.now()}`

  const healthRes = await fetch(`${API}/health`)
  if (!healthRes.ok) throw new Error('Backend not reachable')
  record(steps, '01_stack_health', true, 'backend ok')

  for (const viewport of [
    { name: 'desktop', width: 1440, height: 900 },
    { name: 'mobile', width: 390, height: 844 },
  ]) {
    const browser = await chromium.launch({ headless: true, channel: 'chrome' })
    const page = await browser.newPage({ viewport: { width: viewport.width, height: viewport.height } })

    await page.goto(APP, { waitUntil: 'networkidle', timeout: 60000 })
    await page.waitForSelector('.onboarding-screen', { timeout: 15000 })
    record(steps, `02_${viewport.name}_onboarding_visible`, true, 'onboarding screen shown')

    await page.screenshot({ path: path.join(OUT_DIR, `${viewport.name}_01_welcome.png`) })
    await page.getByRole('button', { name: 'Get started' }).click()

    await page.waitForSelector('text=Tell us about you')
    await page.getByPlaceholder('Your name').fill(demoName)
    await page.locator('.onboarding-form select').first().selectOption('female')
    await page.getByPlaceholder('Birth date (YYYYMMDD)').fill('19950115')
    await page.getByPlaceholder('Height (cm)').fill('168')
    await page.getByRole('button', { name: 'Continue' }).click()
    await page.screenshot({ path: path.join(OUT_DIR, `${viewport.name}_02_profile.png`) })

    await page.waitForSelector('text=What is your health goal?')
    await page.locator('.onboarding-form select').first().selectOption('lose_weight')
    await page.getByPlaceholder('Target weight (optional, e.g. 59 kg)').fill('58 kg')
    await page.getByPlaceholder('Target timeline (optional, e.g. 3 months)').fill('4 months')
    await page.getByRole('button', { name: 'Continue' }).click()
    await page.screenshot({ path: path.join(OUT_DIR, `${viewport.name}_03_goal.png`) })

    await page.waitForSelector('text=Diet preferences')
    await page.getByPlaceholder('Diet preference (e.g. high protein, vegetarian)').fill('high protein')
    await page.getByPlaceholder('Allergies (comma separated)').fill('peanuts')
    await page.getByRole('button', { name: 'Continue' }).click()
    await page.screenshot({ path: path.join(OUT_DIR, `${viewport.name}_04_diet.png`) })

    await page.waitForSelector('text=Your starting weight')
    await page.getByPlaceholder('Current weight (kg)').fill('63.5')
    await page.screenshot({ path: path.join(OUT_DIR, `${viewport.name}_05_weight.png`) })
    await page.getByRole('button', { name: 'Start coaching' }).click()

    await page.waitForSelector('.hero-card h1', { timeout: 20000 })
    await page.waitForFunction(
      (name) => document.querySelector('.hero-card h1')?.textContent?.includes(name),
      demoName,
      { timeout: 15000 }
    )
    const hero = await page.locator('.hero-card h1').textContent()
    const dashboardPass = hero?.includes(demoName)
    record(steps, `03_${viewport.name}_dashboard_after_onboarding`, dashboardPass, hero || '')
    await page.screenshot({ path: path.join(OUT_DIR, `${viewport.name}_06_dashboard.png`) })

    const chatVisible = await page.locator('.chat-section').isVisible()
    record(steps, `04_${viewport.name}_chat_section_visible`, chatVisible, 'main dashboard chat ready')

    await browser.close()
  }

  const users = await (await fetch(`${API}/users`)).json()
  const created = users.find((u) => u.name === demoName)
  record(
    steps,
    '05_user_created_via_api',
    Boolean(created),
    created ? `user_id=${created.user_id} goal=${created.goal} allergies=${JSON.stringify(created.allergies)}` : 'missing'
  )

  if (created) {
    const bundle = await (await fetch(`${API}/users/${created.user_id}`)).json()
    const weightOk = Number(bundle.metrics?.weight_kg) === 63.5
    record(steps, '06_initial_weight_saved', weightOk, `weight=${bundle.metrics?.weight_kg}`)
  } else {
    record(steps, '06_initial_weight_saved', false, 'user not found')
  }

  const overall = steps.every((s) => s.pass)
  const result = {
    entry: '023',
    title: 'Level 2 Onboarding validation',
    date: new Date().toISOString().slice(0, 10),
    overall_pass: overall,
    score: `${steps.filter((s) => s.pass).length}/${steps.length}`,
    steps,
    created_user: created || null,
    screenshot_dir: OUT_DIR,
    notes: [
      'Validation uses ?onboarding=preview because local DB already contains users.',
      'Natural first-time trigger remains users.length === 0 without query param.',
    ],
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
