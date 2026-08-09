#!/usr/bin/env node
/**
 * Entry 022 — Memory Viewer quick re-validation (uses existing user with memory).
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
const USER_ID = process.env.MEMVIEW_USER_ID || '125'

async function api(route) {
  const res = await fetch(`${API}${route}`)
  if (!res.ok) throw new Error(`${route} → ${res.status}`)
  return res.json()
}

function record(steps, key, pass, detail = '') {
  steps.push({ step: key, pass, detail })
  console.log(`${pass ? 'PASS' : 'FAIL'} ${key}: ${detail}`)
}

async function main() {
  fs.mkdirSync(OUT_DIR, { recursive: true })
  const steps = []
  const mem = await api(`/users/${USER_ID}/memory`)
  const sessions = await api(`/users/${USER_ID}/sessions?limit=20`)

  const browser = await chromium.launch({ headless: true, channel: 'chrome' })
  const page = await browser.newPage({ viewport: { width: 1440, height: 900 } })

  await page.goto(APP, { waitUntil: 'networkidle' })
  if (!(await page.locator('.sidebar').isVisible())) {
    await page.getByLabel('Open sidebar').click()
  }
  await page.locator('.sidebar-body select.input').selectOption(USER_ID)
  await page.waitForFunction(
    (id) => document.querySelector('.sidebar-body select.input')?.value === String(id),
    Number(USER_ID),
    { timeout: 10000 }
  )
  record(steps, '01_select_user', true, `user_id=${USER_ID}`)

  await page.getByRole('button', { name: 'View Memory' }).click()
  await page.waitForSelector('.memory-viewer-modal')
  await page.screenshot({ path: path.join(OUT_DIR, '08_selected_user_memory.png') })

  const modal = page.locator('.memory-viewer-modal')
  const profileCards = await modal.locator('.memory-fact-card').count()
  record(steps, '02_profile_context', profileCards >= 3, `cards=${profileCards}`)

  const summaryCards = await modal.locator('.memory-summary-card').count()
  record(steps, '03_session_summaries', summaryCards > 0, `cards=${summaryCards}`)

  const sessionCards = await modal.locator('.memory-session-card').count()
  record(steps, '04_session_history', sessionCards > 0, `cards=${sessionCards}`)

  await modal.getByRole('button', { name: 'View summary' }).first().click()
  await page.waitForTimeout(300)
  const expanded = (await modal.locator('.memory-session-card pre.memory-block-content').count()) > 0
  await modal.getByRole('button', { name: 'Hide summary' }).first().click()
  record(steps, '05_expand_collapse', expanded, `expanded=${expanded}`)

  await modal.getByRole('button', { name: 'Refresh' }).click()
  await page.waitForFunction(() => !document.querySelector('.memory-viewer-loading'), { timeout: 30000 })
  record(steps, '06_refresh', await modal.isVisible(), 'modal stays open after refresh')

  const eligible = sessions.filter((s) => s.status === 'closed' && (s.turn_count ?? 0) >= 2)
  const regenCount = await modal.getByRole('button', { name: 'Regenerate' }).count()
  record(steps, '07_regenerate_eligibility', regenCount === eligible.length, `buttons=${regenCount} eligible=${eligible.length}`)

  await modal.getByLabel('Close memory viewer').click()
  await page.waitForSelector('.memory-viewer-modal', { state: 'detached', timeout: 10000 })
  record(steps, '08_close_modal', true, 'close button works')

  const overall = steps.every((s) => s.pass)
  const result = {
    entry: '022',
    title: 'Level 2 Memory Viewer validation',
    date: new Date().toISOString().slice(0, 10),
    user_id: Number(USER_ID),
    overall_pass: overall,
    score: `${steps.filter((s) => s.pass).length}/${steps.length}`,
    steps,
    api_snapshot: { memory: mem, sessions },
    screenshot_dir: OUT_DIR,
    notes: [
      'Full seed run validated profile, summaries, session history, expand/collapse, refresh, regenerate eligibility.',
      'Minor UX: Escape key does not close Memory Viewer modal (close button and backdrop click work).',
      'Long-term cumulative memory empty state expected for first closed session (rollup threshold not reached).',
    ],
  }
  fs.writeFileSync(RESULT_JSON, JSON.stringify(result, null, 2))
  console.log(`\nResult: ${overall ? 'PASS' : 'FAIL'} (${result.score})`)
  await browser.close()
  process.exit(overall ? 0 : 1)
}

main().catch((err) => {
  console.error(err)
  process.exit(1)
})
