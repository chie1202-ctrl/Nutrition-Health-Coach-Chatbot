import React, { useEffect, useMemo, useRef, useState } from 'react'
import ReactMarkdown from 'react-markdown'

const API_BASE = import.meta.env.VITE_API_BASE_URL || 'http://127.0.0.1:8000'
const FOOD_CHOICE_COMPOSER_PREFIX = 'I am eating out tonight — help me compare takeaway options. '
const FOOD_CHOICE_COMPOSER_JOINER = ' vs '
const FOOD_CHOICE_COMPOSER_SUFFIX = '. Which is better for my profile?'
const ROLLUP_SESSION_THRESHOLD = 3
const SUMMARIZATION_POLL_MS = 2500
const SUMMARIZATION_TIMEOUT_MS = 90000

function MaterialIcon({ name, filled = false, className = '' }) {
  const style = filled ? { fontVariationSettings: "'FILL' 1, 'wght' 400, 'GRAD' 0, 'opsz' 24" } : undefined
  return (
    <span className={`material-symbols-outlined ${className}`.trim()} style={style} aria-hidden="true">
      {name}
    </span>
  )
}

function userInitials(name) {
  const parts = String(name || '').trim().split(/\s+/).filter(Boolean)
  if (!parts.length) return '?'
  return parts.slice(0, 2).map((part) => part[0]).join('').toUpperCase()
}

function scrollToSection(selector, onAfterScroll, { block = 'start' } = {}) {
  const target = document.querySelector(selector)
  const scroller = document.querySelector('.content-wrap')
  if (!target) {
    onAfterScroll?.()
    return
  }
  if (!scroller) {
    target.scrollIntoView({ behavior: 'smooth', block })
    onAfterScroll?.()
    return
  }
  const scrollerRect = scroller.getBoundingClientRect()
  const targetRect = target.getBoundingClientRect()
  const offset = block === 'end'
    ? targetRect.bottom - scrollerRect.bottom
    : targetRect.top - scrollerRect.top
  scroller.scrollTo({ top: scroller.scrollTop + offset, behavior: 'smooth' })
  onAfterScroll?.()
}

function scrollContentToBottom(onAfterScroll, { behavior = 'smooth' } = {}) {
  const scroller = document.querySelector('.content-wrap')
  if (!scroller) {
    onAfterScroll?.()
    return
  }
  const run = () => {
    const anchor = document.querySelector('.chat-input-shell') || document.querySelector('.chat-section')
    if (anchor) {
      const scrollerRect = scroller.getBoundingClientRect()
      const anchorRect = anchor.getBoundingClientRect()
      const alignOffset = anchorRect.bottom - scrollerRect.bottom
      if (alignOffset > 0.5) {
        scroller.scrollBy({ top: alignOffset, behavior })
      }
    }
    const maxTop = Math.max(0, scroller.scrollHeight - scroller.clientHeight)
    if (scroller.scrollTop < maxTop - 1) {
      scroller.scrollTo({ top: maxTop, behavior })
    }
    onAfterScroll?.()
  }
  requestAnimationFrame(() => requestAnimationFrame(run))
}

function scrollToChat(onAfterScroll) {
  scrollContentToBottom(() => {
    const composerOption = document.querySelector('.food-choice-composer-option')
    const chatInput = document.querySelector('.chat-input')
    ;(composerOption || chatInput)?.focus({ preventScroll: true })
    onAfterScroll?.()
  })
}

function buildFoodChoicePrompt(optionA, optionB) {
  return `${FOOD_CHOICE_COMPOSER_PREFIX}${optionA.trim()}${FOOD_CHOICE_COMPOSER_JOINER}${optionB.trim()}${FOOD_CHOICE_COMPOSER_SUFFIX}`
}

function App() {
  const [sidebarOpen, setSidebarOpen] = useState(() => !isMobileViewport())
  const [users, setUsers] = useState([])
  const [selectedUserId, setSelectedUserId] = useState(null)
  const [userBundle, setUserBundle] = useState(null)
  const [chatHistory, setChatHistory] = useState([])
  const [message, setMessage] = useState('')
  const [loading, setLoading] = useState(true)
  const [sending, setSending] = useState(false)
  const sendingRef = useRef(false)
  const [error, setError] = useState('')

  const [showAddUser, setShowAddUser] = useState(false)
  const [showEditUser, setShowEditUser] = useState(false)
  const [showDeleteUser, setShowDeleteUser] = useState(false)
  const [showWeightEntry, setShowWeightEntry] = useState(false)
  const [showWeightChart, setShowWeightChart] = useState(false)
  const [weightWindow, setWeightWindow] = useState(7)
  const [editingWeight, setEditingWeight] = useState(null)

  const [newUser, setNewUser] = useState(emptyUserForm())
  const [editUser, setEditUser] = useState(emptyUserForm())
  const [weightForm, setWeightForm] = useState({ weight_kg: '', recorded_at: '', note: '' })
  const [mealPlan, setMealPlan] = useState(null)
  const [mealPlanDayIndex, setMealPlanDayIndex] = useState(0)
  const [generatingMealPlan, setGeneratingMealPlan] = useState(false)
  const [memoryState, setMemoryState] = useState(null)
  const [closingSession, setClosingSession] = useState(false)
  const [sessionSummarization, setSessionSummarization] = useState(null)
  const [summarizationSlow, setSummarizationSlow] = useState(false)
  const [showSettings, setShowSettings] = useState(false)
  const [showMemoryViewer, setShowMemoryViewer] = useState(false)
  const [showDietPreference, setShowDietPreference] = useState(false)
  const [dietPreferenceDraft, setDietPreferenceDraft] = useState('')
  const [savingDietPreference, setSavingDietPreference] = useState(false)
  const [showAllergies, setShowAllergies] = useState(false)
  const [allergiesDraft, setAllergiesDraft] = useState('')
  const [allergiesOtherDraft, setAllergiesOtherDraft] = useState('')
  const [savingAllergies, setSavingAllergies] = useState(false)
  const [onboardingDismissed, setOnboardingDismissed] = useState(false)
  const [pendingScrollToChat, setPendingScrollToChat] = useState(false)
  const [foodChoiceComposerOpen, setFoodChoiceComposerOpen] = useState(false)
  const [foodChoiceOptionA, setFoodChoiceOptionA] = useState('')
  const [foodChoiceOptionB, setFoodChoiceOptionB] = useState('')
  const foodChoiceOptionARef = useRef(null)
  const [systemHealth, setSystemHealth] = useState(null)
  const onboardingPreview = useMemo(() => readOnboardingPreviewFlag(), [])
  const aiReady = Boolean(systemHealth?.ollama_reachable)
  const showOnboarding = !loading && !onboardingDismissed && (users.length === 0 || onboardingPreview)

  useEffect(() => { bootstrap() }, [])
  useEffect(() => {
    const onResize = () => {
      if (isMobileViewport()) setSidebarOpen(false)
    }
    window.addEventListener('resize', onResize)
    return () => window.removeEventListener('resize', onResize)
  }, [])
  useEffect(() => {
    setSessionSummarization(null)
    setSummarizationSlow(false)
    if (selectedUserId) refreshUserData(selectedUserId)
  }, [selectedUserId])

  useEffect(() => {
    if (!sessionSummarization || !selectedUserId) return undefined

    let cancelled = false

    async function pollSummarization() {
      try {
        const summaries = await api(`/users/${selectedUserId}/summaries?limit=20`)
        if (cancelled) return
        const ready = summaries.some(
          (item) =>
            item.summary_type === 'session' &&
            Number(item.session_id) === Number(sessionSummarization.sessionId)
        )
        if (ready) {
          setSessionSummarization(null)
          setSummarizationSlow(false)
          await refreshUserData(selectedUserId)
        }
      } catch {
        // keep polling until timeout
      }
    }

    pollSummarization()
    const intervalId = window.setInterval(pollSummarization, SUMMARIZATION_POLL_MS)
    const timeoutId = window.setTimeout(() => {
      if (!cancelled) {
        setSessionSummarization(null)
        setSummarizationSlow(true)
      }
    }, SUMMARIZATION_TIMEOUT_MS)

    return () => {
      cancelled = true
      window.clearInterval(intervalId)
      window.clearTimeout(timeoutId)
    }
  }, [sessionSummarization, selectedUserId])
  useEffect(() => {
    if (userBundle?.user) {
      setEditUser(userToForm(userBundle.user, userBundle.metrics))
      setMealPlan(userBundle.meal_plan || null)
      setWeightForm((prev) => ({
        ...prev,
        weight_kg: userBundle.metrics?.weight_kg || '',
        recorded_at: toDateTimeLocal(new Date()),
      }))
    }
  }, [userBundle])

  useEffect(() => {
    setMealPlanDayIndex(0)
  }, [mealPlan, selectedUserId])

  useEffect(() => {
    if (!pendingScrollToChat || !selectedUserId || !userBundle) return
    const timer = window.setTimeout(() => {
      scrollToChat()
      setPendingScrollToChat(false)
    }, 350)
    return () => window.clearTimeout(timer)
  }, [pendingScrollToChat, selectedUserId, userBundle])

  useEffect(() => {
    if (!foodChoiceComposerOpen) return
    const timer = window.setTimeout(() => scrollToChat(), 120)
    return () => window.clearTimeout(timer)
  }, [foodChoiceComposerOpen])

  useEffect(() => {
    const list = document.querySelector('.chat-list')
    if (!list) return
    list.scrollTop = list.scrollHeight
  }, [chatHistory, sending])

  useEffect(() => {
    sendingRef.current = sending
  }, [sending])

  async function refreshSystemHealth() {
    try {
      const health = await api('/health')
      setSystemHealth(health)
    } catch (err) {
      setSystemHealth(null)
    }
  }

  async function bootstrap() {
    setLoading(true)
    setError('')
    try {
      const [usersData] = await Promise.all([api('/users'), refreshSystemHealth()])
      setUsers(usersData)
      if (usersData.length > 0) setSelectedUserId(usersData[0].user_id)
    } catch (err) {
      setError(err.message)
    } finally {
      setLoading(false)
    }
  }

  async function refreshUserData(userId, { skipChatHistory = false } = {}) {
    // Never replace live chat UI while a stream is in flight — meal-plan / memory
    // refreshes would wipe the streaming placeholder and make Q&A "disappear".
    // Use a ref so in-flight callers (started before sending=true) still see the live flag.
    const preserveChat = skipChatHistory || sendingRef.current
    try {
      const [bundle, chats, memory] = await Promise.all([
        api(`/users/${userId}`),
        preserveChat ? Promise.resolve(null) : api(`/users/${userId}/chat`),
        api(`/users/${userId}/memory`),
      ])
      setUserBundle(bundle)
      if (!preserveChat && chats) setChatHistory(normalizeChatHistory(chats))
      setMemoryState(memory)
      setMealPlan(bundle.meal_plan || null)
    } catch (err) {
      setError(err.message)
    }
  }

  async function refreshUsersAndBundle(preferredUserId = null) {
    const usersData = await api('/users')
    setUsers(usersData)
    if (!usersData.length) {
      setSelectedUserId(null)
      setUserBundle(null)
      setChatHistory([])
      setOnboardingDismissed(false)
      return
    }
    const exists = usersData.some((u) => u.user_id === preferredUserId)
    const nextId = exists ? preferredUserId : usersData[0].user_id
    setSelectedUserId(nextId)
    await refreshUserData(nextId)
  }

  async function handleOnboardingComplete(form) {
    setError('')
    const created = await api('/users', {
      method: 'POST',
      body: JSON.stringify(toUserPayload(form)),
    })
    setOnboardingDismissed(true)
    await refreshUsersAndBundle(created.user.user_id)
    setPendingScrollToChat(true)
    if (isMobileViewport()) setSidebarOpen(false)
  }

  async function handleCreateUser(e) {
    e.preventDefault()
    try {
      const created = await api('/users', {
        method: 'POST',
        body: JSON.stringify({
          ...toUserPayload(newUser),
        }),
      })
      setShowAddUser(false)
      setNewUser(emptyUserForm())
      await refreshUsersAndBundle(created.user.user_id)
    } catch (err) {
      setError(err.message)
    }
  }

  async function handleUpdateUser(e) {
    e.preventDefault()
    if (!selectedUserId) return
    try {
      await api(`/users/${selectedUserId}`, {
        method: 'PUT',
        body: JSON.stringify({
          ...toUserPayload(editUser),
        }),
      })
      setShowEditUser(false)
      await refreshUsersAndBundle(selectedUserId)
    } catch (err) {
      setError(err.message)
    }
  }

  function openDietPreferenceModal() {
    setDietPreferenceDraft(normalizeDietPreferenceValue(selectedUser?.diet_preference || ''))
    setShowDietPreference(true)
  }

  async function handleSaveDietPreference(e) {
    e.preventDefault()
    if (!selectedUserId || !selectedUser) return
    setSavingDietPreference(true)
    try {
      await api(`/users/${selectedUserId}`, {
        method: 'PUT',
        body: JSON.stringify({
          ...toUserPayload(userToForm(selectedUser, metrics)),
          diet_preference: dietPreferenceDraft,
        }),
      })
      setShowDietPreference(false)
      await refreshUsersAndBundle(selectedUserId)
    } catch (err) {
      setError(err.message)
    } finally {
      setSavingDietPreference(false)
    }
  }

  function openAllergiesModal() {
    const selected = normalizeAllergyList(parseAllergyList(selectedUser?.allergies))
    const known = new Set(ALLERGY_OPTIONS.map((item) => normalizeAllergyValue(item.value)))
    const preset = selected.filter((item) => known.has(item))
    const custom = selected.filter((item) => !known.has(item))
    setAllergiesDraft(allergyListToCsv(preset))
    setAllergiesOtherDraft(custom.join(', '))
    setShowAllergies(true)
  }

  async function handleSaveAllergies(e) {
    e.preventDefault()
    if (!selectedUserId || !selectedUser) return
    setSavingAllergies(true)
    try {
      const allergies = mergeAllergySelections(allergiesDraft, allergiesOtherDraft)
      await api(`/users/${selectedUserId}`, {
        method: 'PUT',
        body: JSON.stringify({
          ...toUserPayload(userToForm(selectedUser, metrics)),
          allergies,
        }),
      })
      setShowAllergies(false)
      await refreshUsersAndBundle(selectedUserId)
    } catch (err) {
      setError(err.message)
    } finally {
      setSavingAllergies(false)
    }
  }

  async function handleDeleteUser() {
    if (!selectedUserId) return
    try {
      await api(`/users/${selectedUserId}`, { method: 'DELETE' })
      setShowDeleteUser(false)
      await refreshUsersAndBundle(null)
    } catch (err) {
      setError(err.message)
    }
  }

  async function handleUpsertWeight(e) {
    e.preventDefault()
    if (!selectedUserId) return
    try {
      await api(`/users/${selectedUserId}/weight`, {
        method: 'POST',
        body: JSON.stringify({
          user_id: selectedUserId,
          weight_kg: Number(weightForm.weight_kg),
          recorded_at: fromDateTimeLocal(weightForm.recorded_at),
          note: weightForm.note || null,
        }),
      })
      setShowWeightEntry(false)
      await refreshUserData(selectedUserId)
    } catch (err) {
      setError(err.message)
    }
  }

  async function handleUpdateWeightRecord(e) {
    e.preventDefault()
    if (!editingWeight || !selectedUserId) return
    try {
      await api(`/weights/${editingWeight.metric_id}`, {
        method: 'PUT',
        body: JSON.stringify({
          user_id: selectedUserId,
          weight_kg: Number(weightForm.weight_kg),
          recorded_at: fromDateTimeLocal(weightForm.recorded_at),
          note: weightForm.note || null,
        }),
      })
      setEditingWeight(null)
      await refreshUserData(selectedUserId)
    } catch (err) {
      setError(err.message)
    }
  }

  async function handleDeleteWeightRecord(metricId) {
    if (!selectedUserId) return
    try {
      await api(`/weights/${metricId}`, { method: 'DELETE' })
      await refreshUserData(selectedUserId)
    } catch (err) {
      setError(err.message)
    }
  }

  async function handleGenerateMealPlan() {
    if (!selectedUserId || !aiReady || sending) return
    setGeneratingMealPlan(true)
    try {
      const result = await api(`/users/${selectedUserId}/meal-plan`, { method: 'POST' })
      setMealPlan(result)
      // Meal plan does not change chat transcripts; never reload chatHistory here —
      // a concurrent food-choice/chat stream would lose its streaming UI.
      await refreshUserData(selectedUserId, { skipChatHistory: true })
      await refreshSystemHealth()
    } catch (err) {
      setError(err.message)
    } finally {
      setGeneratingMealPlan(false)
    }
  }

  async function handleSendMessage(e, options = {}) {
    e?.preventDefault?.()
    const outgoing = (options.message ?? message).trim()
    if (!selectedUserId || !outgoing || sending || generatingMealPlan || !aiReady) return
    if (!options.message) setMessage('')
    setSending(true)
    const placeholderId = Date.now()
    const priorHistory = chatHistory
    setChatHistory([
      ...priorHistory,
      { role: 'user', content: outgoing, timestamp: new Date().toISOString() },
      { role: 'assistant', content: '', id: placeholderId, streaming: true, timestamp: new Date().toISOString() },
    ])
    try {
      await streamChat(
        {
          user_id: selectedUserId,
          message: outgoing,
          force_new_session: Boolean(options.forceNewSession),
        },
        {
          onMeta: (meta) => {
            if (meta?.sources?.length) {
              setChatHistory((history) =>
                history.map((msg) =>
                  msg.id === placeholderId ? { ...msg, sources: meta.sources } : msg
                )
              )
            }
          },
          onToken: (text) => {
            setChatHistory((history) => {
              if (history.some((msg) => msg.id === placeholderId)) {
                return history.map((msg) =>
                  msg.id === placeholderId ? { ...msg, content: `${msg.content || ''}${text}` } : msg
                )
              }
              // Placeholder was wiped (e.g. by an older refresh race); recreate streaming row.
              return [
                ...history,
                { role: 'assistant', content: text, id: placeholderId, streaming: true, timestamp: new Date().toISOString() },
              ]
            })
          },
          onDone: (result) => {
            const finished = {
              role: 'assistant',
              content: result.reply,
              foodChoice: isFoodChoiceUsable(result.food_choice) ? result.food_choice : null,
              sources: result.sources?.length ? result.sources : [],
              safetyBlocked: Boolean(result.safety_blocked),
              streaming: false,
              timestamp: new Date().toISOString(),
            }
            setChatHistory((history) => {
              if (history.some((msg) => msg.id === placeholderId)) {
                return history.map((msg) =>
                  msg.id === placeholderId
                    ? { ...finished, sources: finished.sources.length ? finished.sources : msg.sources || [] }
                    : msg
                )
              }
              // Recover if streaming placeholder was removed mid-request.
              return [...history, finished]
            })
          },
        }
      )
      await refreshUserData(selectedUserId, { skipChatHistory: true })
      await refreshSystemHealth()
    } catch (err) {
      setError(err.message)
      setChatHistory(priorHistory)
    } finally {
      setSending(false)
    }
  }

  function closeFoodChoiceComposer() {
    setFoodChoiceComposerOpen(false)
    setFoodChoiceOptionA('')
    setFoodChoiceOptionB('')
  }

  function handleCompareTakeaway() {
    if (!selectedUserId || sending || generatingMealPlan || !aiReady) return
    setFoodChoiceOptionA('')
    setFoodChoiceOptionB('')
    setFoodChoiceComposerOpen(true)
  }

  function handleChatSubmit(e) {
    e.preventDefault()
    if (foodChoiceComposerOpen) {
      const optionA = foodChoiceOptionA.trim()
      const optionB = foodChoiceOptionB.trim()
      if (!optionA || !optionB || sending || !aiReady) return
      closeFoodChoiceComposer()
      handleSendMessage(null, { message: buildFoodChoicePrompt(optionA, optionB) })
      return
    }
    handleSendMessage(e)
  }

  const foodChoiceComposerReady = Boolean(foodChoiceOptionA.trim() && foodChoiceOptionB.trim())

  async function handleNewConversation() {
    if (!selectedUserId || closingSession) return
    setClosingSession(true)
    setSummarizationSlow(false)
    try {
      const result = await api(`/users/${selectedUserId}/sessions/close`, { method: 'POST' })
      if (result.summarization_pending && result.session_id != null) {
        setSessionSummarization({ sessionId: result.session_id })
      } else {
        setSessionSummarization(null)
      }
      await refreshUserData(selectedUserId)
    } catch (err) {
      setError(err.message)
      setSessionSummarization(null)
    } finally {
      setClosingSession(false)
    }
  }

  const selectedUser = userBundle?.user
  const metrics = userBundle?.metrics
  const weightSeries = normalizeSeries(metrics?.series || [])
  const sidebarSeries = weightSeries.slice(-5)

  if (loading) return <div className="app-loading">Loading NutriCoachAI...</div>

  if (showOnboarding) {
    return (
      <OnboardingFlow
        preview={onboardingPreview}
        aiReady={aiReady}
        error={error}
        onError={setError}
        onComplete={handleOnboardingComplete}
      />
    )
  }

  return (
    <div className={`app-shell ${sidebarOpen ? 'sidebar-open' : 'sidebar-closed'}`}>
      {sidebarOpen && <div className="sidebar-backdrop" onClick={() => setSidebarOpen(false)} aria-hidden="true" />}
      {sidebarOpen && (
        <aside className="sidebar">
          <div className="sidebar-top">
            <div className="brand-stack">
              <div className="user-avatar-chip">{userInitials(selectedUser?.name)}</div>
              <div>
                <div className="brand-title-row">
                  <div className="brand-title">{selectedUser?.name || 'NutriCoachAI'}</div>
                  {aiReady ? <span className="dot-online" /> : null}
                </div>
                <div className="brand-subtitle">Local health coaching</div>
              </div>
            </div>
            <button type="button" className="icon-button ghost" onClick={() => setSidebarOpen(false)} aria-label="Close sidebar">
              <MaterialIcon name="close" />
            </button>
          </div>

          <div className="sidebar-body">
            <nav className="sidebar-nav" aria-label="Dashboard sections">
              <button type="button" className="sidebar-nav-link" onClick={() => { scrollToSection('.hero-card', () => { if (isMobileViewport()) setSidebarOpen(false) }) }}>
                <MaterialIcon name="dashboard" />
                <span>Dashboard</span>
              </button>
              <button type="button" className="sidebar-nav-link" onClick={() => { scrollToSection('.plan-section', () => { if (isMobileViewport()) setSidebarOpen(false) }) }}>
                <MaterialIcon name="restaurant" />
                <span>Meal Plan</span>
              </button>
              <button type="button" className="sidebar-nav-link" onClick={() => { scrollToSection('.goal-progress-section', () => { if (isMobileViewport()) setSidebarOpen(false) }) }}>
                <MaterialIcon name="insights" />
                <span>Progress</span>
              </button>
              <button type="button" className="sidebar-nav-link" onClick={() => { scrollToChat(() => { if (isMobileViewport()) setSidebarOpen(false) }) }}>
                <MaterialIcon name="psychology" />
                <span>Coach</span>
              </button>
            </nav>

            <SectionTitle icon="person" title="USER PROFILE" />
            <select className="input select" value={selectedUserId ?? ''} onChange={(e) => setSelectedUserId(Number(e.target.value))}>
              {users.map((user) => <option key={user.user_id} value={user.user_id}>{user.name}</option>)}
            </select>
            <div className="button-stack-2">
              <button className="cta-button" onClick={() => setShowAddUser(true)}>Add New User</button>
              <button className="secondary-button" onClick={() => setShowEditUser(true)} disabled={!selectedUser}>Edit User</button>
            </div>
            <button className="danger-button" onClick={() => setShowDeleteUser(true)} disabled={!selectedUser}>Delete User</button>

            <SectionTitle icon="monitoring" title="CURRENT METRICS" />
            <MetricCard title="WEIGHT" value={metrics?.weight_kg ? `${metrics.weight_kg} kg` : '--'} icon="fitness_center" />
            <button className="cta-button" onClick={() => { setEditingWeight(null); setShowWeightEntry(true) }}>Update Weight</button>
            <MetricCard title="BMI" value={metrics?.bmi ?? '--'} icon="monitor_weight" badge={metrics?.bmi_label === 'Normal' ? '✓ Normal' : metrics?.bmi_label} />
            <MetricCard title="EST. REE" value={metrics?.ree ? `${metrics.ree} kcal/day` : '--'} subtitle="Estimated Resting Energy Expenditure" icon="bolt" />

            <SectionTitle icon="trending_up" title="WEIGHT PROGRESS" />
            <SidebarWeightChart series={sidebarSeries} onClick={() => setShowWeightChart(true)} />

            <div className="sidebar-privacy-card">
              <h4>Privacy-first</h4>
              <p>Your health data stays on this device with local Ollama inference — no cloud LLM API.</p>
            </div>
          </div>
        </aside>
      )}

      <main className="page">
        <header className="topbar">
          <div className="topbar-left">
            {!sidebarOpen ? (
              <button type="button" className="icon-button solid" onClick={() => setSidebarOpen(true)} aria-label="Open sidebar">
                <MaterialIcon name="menu" />
              </button>
            ) : null}
          </div>
          <div className="topbar-brand">
            <div className="mini-brand-mark">N</div>
            <div className="mini-brand-name">NutriCoachAI</div>
          </div>
          <div className="topbar-right">
            <button
              type="button"
              className="icon-button settings"
              aria-label="Settings and help"
              onClick={() => {
                refreshSystemHealth()
                setShowSettings(true)
              }}
            >
              <MaterialIcon name="settings" />
            </button>
          </div>
        </header>

        <div className="content-wrap">
          {error && <div className="error-banner">{error}</div>}
          {systemHealth && !aiReady && (
            <div className="warning-banner">
              Local AI engine (Ollama) is not running. Start Ollama to use chat and meal-plan generation.
              Run <code>ollama serve</code> or use <code>./start.sh</code> from the project root.
            </div>
          )}
          <section className="hero-card">
            <h1>Coaching {selectedUser?.name || 'User'}</h1>
            <div className="hero-stats">
              <HeroMetric value={metrics?.weight_kg ?? '--'} suffix="kg" label="WEIGHT" icon="fitness_center" />
              <HeroMetric
                value={metrics?.bmi ?? '--'}
                label={metrics?.bmi_label ? `BMI · ${metrics.bmi_label}` : 'BMI'}
                icon="monitor_weight"
              />
              <HeroMetric value={metrics?.ree ?? '--'} suffix="kcal" label="ESTIMATED REE" icon="bolt" />
            </div>
          </section>

          <GoalProgressCard user={selectedUser} metrics={metrics} weightSeries={weightSeries} onEditGoal={() => setShowEditUser(true)} />

          <p className="medical-disclaimer medical-disclaimer-sr" role="note" aria-hidden="true">
            NutriCoachAI provides general wellness coaching only — not medical advice, diagnosis, or treatment.
            Always consult a qualified healthcare professional for personal medical concerns.
          </p>

          <section className="plan-section bento-card">
            <div className="plan-header-row">
              <div>
                <div className="plan-title-row">
                  <h2>
                    7-Day Meal Plan
                    {mealPlan?.plan?.days?.length > 0 && typeof mealPlan.llm_degraded === 'boolean' ? (
                      <span className={`plan-source-badge ${mealPlan.llm_degraded ? 'template' : 'ai'}`}>
                        {mealPlan.llm_degraded ? 'Template Fallback' : 'AI Generated'}
                      </span>
                    ) : null}
                  </h2>
                  <div className="profile-tags profile-tags-inline">
                    {selectedUser?.goal ? <span className="profile-chip">Goal: {formatGoalLabel(selectedUser.goal)}</span> : null}
                    <button
                      type="button"
                      className="profile-chip profile-chip-button"
                      onClick={openDietPreferenceModal}
                      disabled={!selectedUser}
                      aria-label="Set diet preference"
                    >
                      Diet: {formatDietPreferenceLabel(selectedUser?.diet_preference)}
                      <MaterialIcon name="edit" className="profile-chip-icon" />
                    </button>
                    <button
                      type="button"
                      className="profile-chip profile-chip-button"
                      onClick={openAllergiesModal}
                      disabled={!selectedUser}
                      aria-label="Set allergies"
                    >
                      Allergies: {formatAllergiesLabel(selectedUser?.allergies)}
                      <MaterialIcon name="edit" className="profile-chip-icon" />
                    </button>
                    {selectedUser?.activity_level ? <span className="profile-chip">Activity: {formatActivityLabel(selectedUser.activity_level)}</span> : null}
                    {selectedUser?.medical_conditions?.map((item) => <span key={item} className="profile-chip">{item}</span>)}
                  </div>
                </div>
                <p>Personalized to the selected user's goals, conditions, and preferences.</p>
              </div>
              <div className="plan-header-actions">
              <button
                className="cta-button plan-generate-button"
                onClick={handleGenerateMealPlan}
                disabled={!selectedUser || generatingMealPlan || sending || !aiReady}
                title={
                  !aiReady
                    ? 'Start Ollama to generate a meal plan'
                    : sending
                      ? 'Wait for the current coach reply to finish'
                      : undefined
                }
              >
                <MaterialIcon name="auto_awesome" />
                {generatingMealPlan ? 'Generating...' : aiReady ? 'Generate Meal Plan' : 'AI Engine Required'}
              </button>
              <button
                className="secondary-button plan-compare-button"
                type="button"
                onClick={handleCompareTakeaway}
                disabled={!selectedUser || sending || generatingMealPlan || !aiReady}
                title={
                  !aiReady
                    ? 'Start Ollama to compare takeaway options'
                    : generatingMealPlan
                      ? 'Wait for meal-plan generation to finish'
                      : undefined
                }
              >
                <MaterialIcon name="compare_arrows" />
                Compare takeaway options
              </button>
              </div>
            </div>
            {mealPlan?.plan ? (
              <>
                <div className="plan-summary-card">{mealPlan.plan.summary}</div>
                <MealPlanNutritionTargets targets={mealPlan.plan.nutrition_targets} />
                <MealPlanDayView
                  days={mealPlan.plan.days}
                  dayIndex={mealPlanDayIndex}
                  onDayChange={setMealPlanDayIndex}
                />
              </>
            ) : (
              <div className="plan-empty-card">No meal plan generated yet. Create one based on the saved profile.</div>
            )}
          </section>

          <section className="chat-section chat-panel bento-card">
          <div className="chat-panel-header">
            <div className="chat-coach-avatar">
              <MaterialIcon name="psychology" filled />
            </div>
            <div className="chat-panel-header-text">
              <h2>Chat with your Coach</h2>
              <div className="chat-status-row">
                <span className="chat-status-dot" />
                <p>{aiReady ? 'Local AI active' : 'AI engine offline'}</p>
              </div>
            </div>
            <div className="chat-heading-actions">
              <button
                className="secondary-button memory-view-button"
                type="button"
                onClick={() => setShowMemoryViewer(true)}
                disabled={!selectedUser}
              >
                View Memory
              </button>
              <button
                className="secondary-button chat-new-session-button"
                type="button"
                onClick={handleNewConversation}
                disabled={!selectedUser || closingSession}
              >
                {closingSession ? 'Closing...' : 'New Conversation'}
              </button>
            </div>
          </div>
          <div className="chat-heading-row" hidden aria-hidden="true">
            <span className="chat-heading-dot" />
            <div>
              <h2>Chat with your Coach</h2>
              <p>
                AI-powered · Local & Private · Responds in your language
                {memoryState?.cumulative_summary || memoryState?.recent_session_summaries?.length
                  ? ' · Coach remembers your progress'
                  : ''}
              </p>
            </div>
          </div>
          <div className="chat-panel-body">
            {sessionSummarization ? (
              <div className="info-banner summarization-pending-banner" role="status" aria-live="polite">
                Summarizing your last conversation… Memory recall in new chats may improve once this
                finishes (usually 10–30 seconds). You can keep chatting in the meantime.
              </div>
            ) : null}
            {summarizationSlow ? (
              <div className="info-banner summarization-slow-banner" role="status">
                Session summary is taking longer than usual. Check View Memory or try New Conversation
                again after a moment if recall seems incomplete.
              </div>
            ) : null}
            <div className="chat-list">
              {chatHistory.length === 0 && selectedUser && (
                <ChatMessage role="assistant" content={`Hi ${selectedUser.name}! 👋\n\nYour latest stats — Weight: ${metrics?.weight_kg ?? '--'} kg · BMI: ${metrics?.bmi ?? '--'} · Estimated REE: ${metrics?.ree ?? '--'} kcal/day\n\nWhat would you like to work on today?`} />
              )}
              {chatHistory.map((item, index) => (
                <ChatMessage
                  key={`${item.timestamp}-${index}`}
                  role={item.role}
                  content={item.content}
                  foodChoice={item.foodChoice}
                  sources={item.sources}
                  safetyBlocked={item.safetyBlocked}
                  streaming={item.streaming}
                />
              ))}
            </div>
            <form className={`chat-input-shell${foodChoiceComposerOpen ? ' chat-input-shell-composer' : ''}`} onSubmit={handleChatSubmit}>
              {foodChoiceComposerOpen ? (
                <>
                  <div className="food-choice-composer" role="group" aria-label="Compare takeaway options">
                    <span className="food-choice-composer-text">{FOOD_CHOICE_COMPOSER_PREFIX}</span>
                    <input
                      ref={foodChoiceOptionARef}
                      className="food-choice-composer-option"
                      type="text"
                      placeholder="Option A"
                      value={foodChoiceOptionA}
                      onChange={(e) => setFoodChoiceOptionA(e.target.value)}
                      disabled={!aiReady || sending || generatingMealPlan}
                      aria-label="First meal option"
                    />
                    <span className="food-choice-composer-text">{FOOD_CHOICE_COMPOSER_JOINER}</span>
                    <input
                      className="food-choice-composer-option"
                      type="text"
                      placeholder="Option B"
                      value={foodChoiceOptionB}
                      onChange={(e) => setFoodChoiceOptionB(e.target.value)}
                      disabled={!aiReady || sending || generatingMealPlan}
                      aria-label="Second meal option"
                    />
                    <span className="food-choice-composer-text">{FOOD_CHOICE_COMPOSER_SUFFIX}</span>
                  </div>
                  <button
                    className="food-choice-composer-close"
                    type="button"
                    onClick={closeFoodChoiceComposer}
                    disabled={sending}
                    aria-label="Cancel comparison"
                    title="Cancel"
                  >
                    <MaterialIcon name="close" />
                  </button>
                  <button
                    className="send-button"
                    disabled={sending || generatingMealPlan || !aiReady || !foodChoiceComposerReady}
                    type="submit"
                    title={
                      !aiReady
                        ? 'Start Ollama first'
                        : generatingMealPlan
                          ? 'Wait for meal-plan generation to finish'
                          : undefined
                    }
                  >
                    <MaterialIcon name="send" />
                  </button>
                </>
              ) : (
                <>
                  <input
                    className="chat-input"
                    placeholder={
                      generatingMealPlan
                        ? 'Meal plan is generating — chat resumes when it finishes...'
                        : aiReady
                          ? 'Ask your coach...'
                          : 'Start Ollama to chat with your coach...'
                    }
                    value={message}
                    onChange={(e) => setMessage(e.target.value)}
                    disabled={!aiReady || sending || generatingMealPlan}
                  />
                  <button
                    className="send-button"
                    disabled={sending || generatingMealPlan || !aiReady || !message.trim()}
                    type="submit"
                    title={
                      !aiReady
                        ? 'Start Ollama first'
                        : generatingMealPlan
                          ? 'Wait for meal-plan generation to finish'
                          : undefined
                    }
                  >
                    <MaterialIcon name="send" />
                  </button>
                </>
              )}
            </form>
          </div>
        </section>
        </div>
      </main>

      <nav className="mobile-bottom-nav" aria-label="Mobile navigation">
        <button type="button" onClick={() => scrollToSection('.hero-card')}>
          <MaterialIcon name="home" filled />
          <span>Home</span>
        </button>
        <button type="button" onClick={() => scrollToSection('.plan-section')}>
          <MaterialIcon name="menu_book" />
          <span>Plan</span>
        </button>
        <button type="button" onClick={() => scrollToChat()}>
          <MaterialIcon name="chat_bubble" />
          <span>Coach</span>
        </button>
        <button type="button" onClick={() => scrollToSection('.goal-progress-section')}>
          <MaterialIcon name="query_stats" />
          <span>Stats</span>
        </button>
      </nav>

      {showAddUser && (
        <Modal title="Add New User" onClose={() => setShowAddUser(false)}>
          <form className="modal-form" onSubmit={handleCreateUser}>
            <UserProfileFields value={newUser} onChange={setNewUser} includeWeight />
            <button className="cta-button submit" type="submit">Create User</button>
          </form>
        </Modal>
      )}

      {showEditUser && selectedUser && (
        <Modal title="Edit User" onClose={() => setShowEditUser(false)}>
          <form className="modal-form" onSubmit={handleUpdateUser}>
            <UserProfileFields value={editUser} onChange={setEditUser} includeWeight />
            <button className="cta-button submit" type="submit">Save Changes</button>
          </form>
        </Modal>
      )}

      {showDietPreference && selectedUser && (
        <Modal title="Diet Preference" onClose={() => setShowDietPreference(false)}>
          <form className="modal-form" onSubmit={handleSaveDietPreference}>
            <p className="modal-lead">Choose a diet style for meal plan generation. Regenerate your plan after saving to apply the change.</p>
            <DietPreferenceSelect value={dietPreferenceDraft} onChange={setDietPreferenceDraft} />
            <button className="cta-button submit" type="submit" disabled={savingDietPreference}>
              {savingDietPreference ? 'Saving...' : 'Save Diet Preference'}
            </button>
          </form>
        </Modal>
      )}

      {showAllergies && selectedUser && (
        <Modal title="Allergies" onClose={() => setShowAllergies(false)}>
          <form className="modal-form" onSubmit={handleSaveAllergies}>
            <p className="modal-lead">Select all allergens to avoid in meal plans and coaching. Regenerate your plan after saving to apply the change.</p>
            <AllergyMultiSelect value={allergiesDraft} onChange={setAllergiesDraft} id="plan-allergies" />
            <input
              className="input"
              placeholder="Other allergies (comma separated, optional)"
              value={allergiesOtherDraft}
              onChange={(e) => setAllergiesOtherDraft(e.target.value)}
            />
            <button className="cta-button submit" type="submit" disabled={savingAllergies}>
              {savingAllergies ? 'Saving...' : 'Save Allergies'}
            </button>
          </form>
        </Modal>
      )}

      {showDeleteUser && selectedUser && (
        <ConfirmModal title="Delete User" body={`Delete ${selectedUser.name} and all related weight/chat history?`} confirmLabel="Delete User" onCancel={() => setShowDeleteUser(false)} onConfirm={handleDeleteUser} />
      )}

      {showWeightEntry && (
        <Modal title={editingWeight ? 'Edit Weight Record' : 'Update Weight'} onClose={() => { setShowWeightEntry(false); setEditingWeight(null) }}>
          <form className="modal-form" onSubmit={editingWeight ? handleUpdateWeightRecord : handleUpsertWeight}>
            <input className="input" placeholder="Weight (kg)" type="number" value={weightForm.weight_kg} onChange={(e) => setWeightForm({ ...weightForm, weight_kg: e.target.value })} />
            <input className="input" type="datetime-local" value={weightForm.recorded_at} onChange={(e) => setWeightForm({ ...weightForm, recorded_at: e.target.value })} />
            <input className="input" placeholder="Note (optional)" value={weightForm.note} onChange={(e) => setWeightForm({ ...weightForm, note: e.target.value })} />
            <button className="cta-button submit" type="submit">{editingWeight ? 'Save Weight Record' : 'Save Weight'}</button>
          </form>
        </Modal>
      )}

      {showSettings && (
        <SettingsAboutModal
          health={systemHealth}
          onRefresh={refreshSystemHealth}
          onClose={() => setShowSettings(false)}
        />
      )}

      {showMemoryViewer && selectedUser && (
        <MemoryViewerModal
          userId={selectedUserId}
          user={selectedUser}
          metrics={metrics}
          aiReady={aiReady}
          onClose={() => setShowMemoryViewer(false)}
          onMemoryUpdated={setMemoryState}
        />
      )}

      {showWeightChart && (
        <WeightProgressModal
          name={selectedUser?.name || 'User'}
          series={weightSeries}
          windowDays={weightWindow}
          onWindowChange={setWeightWindow}
          onClose={() => setShowWeightChart(false)}
          onEdit={(record) => {
            setShowWeightChart(false)
            setEditingWeight(record)
            setWeightForm({ weight_kg: record.weight_kg, recorded_at: toDateTimeLocal(parseRecordDate(record.record_date || record.recorded_at)), note: record.note || '' })
            setTimeout(() => setShowWeightEntry(true), 0)
          }}
          onDelete={handleDeleteWeightRecord}
        />
      )}
    </div>
  )
}

function SectionTitle({ icon, title }) {
  return (
    <div className="section-title">
      <MaterialIcon name={icon} />
      {title}
    </div>
  )
}

function MetricCard({ icon, title, value, subtitle, badge }) {
  return (
    <div className="metric-card">
      <div className="metric-title">
        <MaterialIcon name={icon} />
        {title}
      </div>
      <div className="metric-value">
        {value}
        {badge ? <span className="metric-badge">{badge}</span> : null}
      </div>
      {subtitle ? <div className="metric-subtitle">{subtitle}</div> : null}
    </div>
  )
}

function HeroMetric({ value, suffix, label, icon }) {
  return (
    <div className="hero-metric">
      <div className="hero-metric-content">
        <div className="hero-label">{label}</div>
        <div className="hero-value">
          {value} {suffix ? <span>{suffix}</span> : null}
        </div>
      </div>
      {icon ? (
        <div className="hero-metric-icon">
          <MaterialIcon name={icon} />
        </div>
      ) : null}
    </div>
  )
}

function MealPlanNutritionTargets({ targets }) {
  const formatted = formatNutritionBlock(targets)
  if (!formatted) return null
  return (
    <div className="plan-nutrition-targets" aria-label="Weekly nutrition targets">
      <div className="plan-nutrition-targets-label">Weekly targets</div>
      <div className="plan-nutrition-targets-values">{formatted}</div>
    </div>
  )
}

function formatNutritionBlock(block) {
  if (!block || typeof block !== 'object') return ''
  const parts = []
  if (block.calories != null) parts.push(`${block.calories} kcal`)
  if (block.protein_g != null) parts.push(`Protein ${block.protein_g} g`)
  if (block.carbs_g != null) parts.push(`Carbs ${block.carbs_g} g`)
  if (block.fat_g != null) parts.push(`Fat ${block.fat_g} g`)
  return parts.join(' · ')
}

function MealPlanDayView({ days, dayIndex, onDayChange }) {
  const safeDays = Array.isArray(days) ? days : []
  const totalDays = safeDays.length
  const clampedIndex = totalDays > 0 ? Math.min(Math.max(dayIndex, 0), totalDays - 1) : 0
  const day = safeDays[clampedIndex]

  if (!day) return null

  return (
    <div className="meal-plan-day-shell">
      <div className="meal-card" aria-live="polite">
        <div className="meal-day">Day {day.day}</div>
        <div className="meal-focus">{day.focus}</div>
        <div className="meal-row"><strong>Breakfast:</strong> {day.breakfast}</div>
        <div className="meal-row"><strong>Lunch:</strong> {day.lunch}</div>
        <div className="meal-row"><strong>Dinner:</strong> {day.dinner}</div>
        <div className="meal-row"><strong>Snack:</strong> {day.snack}</div>
        {formatNutritionBlock(day.daily_totals) ? (
          <div className="meal-day-totals">
            <strong>Daily totals:</strong> {formatNutritionBlock(day.daily_totals)}
          </div>
        ) : null}
        {day.notes ? <div className="meal-note">{day.notes}</div> : null}
      </div>
      {totalDays > 1 ? (
        <nav className="meal-plan-nav" aria-label="Meal plan day navigation">
          <button
            type="button"
            className="meal-plan-nav-button"
            onClick={() => onDayChange(clampedIndex - 1)}
            disabled={clampedIndex <= 0}
            aria-label="Previous day"
          >
            <MaterialIcon name="chevron_left" />
            Previous
          </button>
          <span className="meal-plan-nav-status">Day {clampedIndex + 1} of {totalDays}</span>
          <button
            type="button"
            className="meal-plan-nav-button"
            onClick={() => onDayChange(clampedIndex + 1)}
            disabled={clampedIndex >= totalDays - 1}
            aria-label="Next day"
          >
            Next
            <MaterialIcon name="chevron_right" />
          </button>
        </nav>
      ) : null}
    </div>
  )
}

function GoalProgressCard({ user, metrics, weightSeries, onEditGoal }) {
  const progress = buildGoalProgress(user, metrics, weightSeries)
  if (!progress) {
    return (
      <section className="goal-progress-section">
        <div className="goal-progress-card goal-progress-empty">
          <div>
            <h2>Goal Progress</h2>
            <p>Add a target weight in your profile to track progress toward your coaching goal.</p>
          </div>
          <button type="button" className="secondary-button" onClick={onEditGoal}>Edit profile</button>
        </div>
      </section>
    )
  }

  return (
    <section className="goal-progress-section" aria-label="Goal progress">
      <div className="goal-progress-card">
        <div className="goal-progress-header">
          <div>
            <h2>Goal Progress</h2>
            <p>
              {progress.goalLabel}
              {progress.targetTimeline ? ` · ${progress.targetTimeline}` : ''}
            </p>
          </div>
          <div className="goal-progress-percent">{progress.progressPercent}%</div>
        </div>

        <div className="goal-progress-bar-shell" aria-hidden="true">
          <div className="goal-progress-bar-track">
            <div className="goal-progress-bar-fill" style={{ width: `${progress.progressPercent}%` }} />
          </div>
        </div>

        <div className="goal-progress-stats">
          <div className="goal-progress-stat">
            <div className="goal-progress-stat-label">Current</div>
            <div className="goal-progress-stat-value">{formatKg(progress.currentKg)}</div>
          </div>
          <div className="goal-progress-stat">
            <div className="goal-progress-stat-label">Target</div>
            <div className="goal-progress-stat-value">{formatKg(progress.targetKg)}</div>
          </div>
          <div className="goal-progress-stat">
            <div className="goal-progress-stat-label">{progress.remainingLabel}</div>
            <div className="goal-progress-stat-value">{formatKg(progress.remainingKg)}</div>
          </div>
        </div>

        <div className="goal-progress-trend">{progress.trendMessage}</div>
      </div>
    </section>
  )
}

function FoodChoiceCard({ comparison }) {
  if (!isFoodChoiceUsable(comparison)) return null
  const dimensions = [
    { key: 'protein', label: 'Protein' },
    { key: 'carbs', label: 'Carbs' },
    { key: 'sodium', label: 'Sodium' },
    { key: 'glycemic', label: 'Glycemic impact' },
  ]

  return (
    <div className="food-choice-card" aria-label="Meal comparison">
      <div className="food-choice-header">
        <div className="food-choice-option food-choice-option-a">
          <span className="food-choice-option-label">Option A</span>
          <strong>{comparison.option_a}</strong>
        </div>
        <div className="food-choice-vs">vs</div>
        <div className="food-choice-option food-choice-option-b">
          <span className="food-choice-option-label">Option B</span>
          <strong>{comparison.option_b}</strong>
        </div>
      </div>
      <table className="food-choice-table">
        <thead>
          <tr>
            <th scope="col">Dimension</th>
            <th scope="col">{comparison.option_a}</th>
            <th scope="col">{comparison.option_b}</th>
          </tr>
        </thead>
        <tbody>
          {dimensions.map(({ key, label }) => {
            const row = comparison.comparison?.[key] || {}
            return (
              <tr key={key}>
                <th scope="row">{label}</th>
                <td>{row.option_a || '—'}</td>
                <td>{row.option_b || '—'}</td>
              </tr>
            )
          })}
        </tbody>
      </table>
      {comparison.recommendation ? (
        <div className="food-choice-recommendation">
          <strong>Recommendation</strong>
          <p>{comparison.recommendation}</p>
        </div>
      ) : null}
      {comparison.portion_tip ? (
        <div className="food-choice-tip">
          <strong>Portion tip</strong>
          <p>{comparison.portion_tip}</p>
        </div>
      ) : null}
      {comparison.swap_suggestion ? (
        <div className="food-choice-tip">
          <strong>Swap idea</strong>
          <p>{comparison.swap_suggestion}</p>
        </div>
      ) : null}
      {comparison.profile_notes?.length ? (
        <ul className="food-choice-profile-notes">
          {comparison.profile_notes.map((note) => (
            <li key={note}>{note}</li>
          ))}
        </ul>
      ) : null}
    </div>
  )
}

function ChatMessage({ role, content, foodChoice = null, sources = [], safetyBlocked = false, streaming = false }) {
  const roleText = String(role || '').toLowerCase()
  const isAssistant = ['assistant', 'ai', 'model', 'bot', 'coach'].some((token) => roleText.includes(token))
  const isUser = !isAssistant
  const displayContent = content || (streaming ? ' ' : '')
  return (
    <div className={`message-row ${isUser ? 'user' : 'assistant'}`}>
      {!isUser && (
        <div className="avatar assistant-avatar">
          <MaterialIcon name="smart_toy" />
        </div>
      )}
      <div className={`message-bubble ${isUser ? 'user-bubble' : 'assistant-bubble'}${streaming ? ' streaming' : ''}`}>
        {isAssistant ? (
          <div className="message-markdown">
            <ReactMarkdown>{displayContent}</ReactMarkdown>
          </div>
        ) : (
          displayContent
        )}
        {isAssistant && !streaming && foodChoice ? <FoodChoiceCard comparison={foodChoice} /> : null}
        {isAssistant && safetyBlocked ? <div className="safety-notice">Safety guardrail response</div> : null}
        {isAssistant && !streaming && sources?.length > 0 ? (
          <div className="source-citations">
            <span className="source-label">Sources:</span>
            {sources.map((name) => (
              <span key={name} className="source-chip">{name}</span>
            ))}
          </div>
        ) : null}
      </div>
      {isUser && (
        <div className="avatar user-avatar">
          <MaterialIcon name="person" />
        </div>
      )}
    </div>
  )
}

function SidebarWeightChart({ series, onClick }) {
  const chart = buildSidebarChart(series)
  return (
    <button className="chart-card chart-button" onClick={onClick} type="button">
      {series.length ? (
        <svg viewBox="0 0 244 152" className="chart-svg chart-svg-sidebar">
          {chart.yTicks.map((tick) => (
            <g key={`y-${tick.value}`}>
              <line x1={chart.area.left} y1={tick.y} x2={chart.area.right} y2={tick.y} className="grid-line" />
              <text x={chart.area.left - 8} y={tick.y + 4} className="axis-label axis-y" textAnchor="end">{tick.value}</text>
            </g>
          ))}
          <text x={10} y={18} className="axis-title axis-title-small">kg</text>
          {chart.xTicks.map((tick) => (
            <text key={`x-${tick.label}-${tick.x}`} x={tick.x} y={144} className="axis-label axis-x" textAnchor="middle">{tick.label}</text>
          ))}
          {chart.path && <path d={chart.path} className="line-path" />}
          {chart.points.map((point) => (
            <circle key={`${point.dateKey}-${point.x}`} cx={point.x} cy={point.y} r="3.8" className="point-dot" />
          ))}
        </svg>
      ) : <div className="chart-empty">No weight history yet.</div>}
      <div className="chart-footnote">Latest 5 weight records</div>
    </button>
  )
}

function WeightProgressModal({ name, series, windowDays, onWindowChange, onClose, onEdit, onDelete }) {
  const filtered = filterSeriesByDays(series, windowDays)
  const chart = buildChart(filtered, { width: 780, height: 380, padTop: 22, padBottom: 44, padLeft: 64, padRight: 24, xTickMode: 'range', showPointDots: true })
  const statsSource = filtered.length ? filtered : series
  const weights = statsSource.map((item) => item.weight_kg)
  const latest = statsSource.at(-1)?.weight_kg
  const highest = weights.length ? Math.max(...weights) : null
  const lowest = weights.length ? Math.min(...weights) : null
  return (
    <div className="modal-backdrop" onClick={onClose}>
      <div className="modal-card modal-card-wide" onClick={(e) => e.stopPropagation()}>
        <div className="modal-head"><h3>{name} Weight Progress</h3><button className="modal-close" onClick={onClose}>×</button></div>
        <div className="window-switcher">{[7, 14, 30].map((days) => <button key={days} type="button" className={`window-chip ${windowDays === days ? 'active' : ''}`} onClick={() => onWindowChange(days)}>Last {days} days</button>)}</div>
        <div className="summary-grid"><SummaryCard label="Latest" value={latest != null ? `${latest} kg` : '--'} /><SummaryCard label="Highest" value={highest != null ? `${highest} kg` : '--'} /><SummaryCard label="Lowest" value={lowest != null ? `${lowest} kg` : '--'} /></div>
        <div className="modal-chart-shell">
          {filtered.length ? (
            <svg viewBox="0 0 780 380" className="chart-svg chart-svg-modal">
              {chart.yTicks.map((tick) => <g key={`y-${tick.value}`}><line x1={chart.area.left} y1={tick.y} x2={chart.area.right} y2={tick.y} className="grid-line" /><text x={chart.area.left - 10} y={tick.y + 4} className="axis-label axis-y" textAnchor="end">{tick.value}</text></g>)}
              <text x={chart.area.left} y={18} className="axis-title">Weight (kg)</text>
              {chart.xTicks.map((tick) => <g key={`${tick.label}-${tick.x}`}><line x1={tick.x} y1={chart.area.bottom} x2={tick.x} y2={chart.area.bottom + 6} className="axis-tick" /><text x={tick.x} y={chart.area.bottom + 24} className="axis-label axis-x" textAnchor="middle">{tick.label}</text></g>)}
              {chart.path && <path d={chart.path} className="line-path modal-line-path" />}
              {chart.points.map((point) => <g key={`${point.dateKey}-${point.x}`}><circle cx={point.x} cy={point.y} r={point === chart.latestPoint ? '6' : '4.2'} className="point-dot modal-point-dot" /><title>{`${formatLongDate(point.date)} · ${point.weight_kg} kg`}</title></g>)}
            </svg>
          ) : <div className="chart-empty large">No weight records available for this range.</div>}
        </div>
        <div className="modal-note">X-axis shows actual record dates for the selected range. Y-axis shows weight in kilograms.</div>
        <div className="history-table-shell">
          <table className="history-table">
            <thead><tr><th>Date</th><th>Weight</th><th>BMI</th><th>Est. REE</th><th>Actions</th></tr></thead>
            <tbody>
              {filtered.map((record) => (
                <tr key={record.metric_id}>
                  <td>{formatLongDate(record.date)}</td>
                  <td>{record.weight_kg} kg</td>
                  <td>{record.bmi}</td>
                  <td>{record.ree} kcal/day</td>
                  <td className="actions-cell"><button className="table-button" onClick={() => onEdit(record)}>Edit</button><button className="table-button danger" onClick={() => onDelete(record.metric_id)}>Delete</button></td>
                </tr>
              ))}
            </tbody>
          </table>
        </div>
      </div>
    </div>
  )
}


const DIET_PREFERENCE_OPTIONS = [
  { value: '', label: 'No specific preference' },
  { value: 'balanced', label: 'Balanced' },
  { value: 'high_protein', label: 'High protein' },
  { value: 'vegetarian', label: 'Vegetarian' },
  { value: 'vegan', label: 'Vegan' },
  { value: 'low_carb', label: 'Low carb' },
  { value: 'mediterranean', label: 'Mediterranean' },
  { value: 'pescatarian', label: 'Pescatarian' },
]

function normalizeDietPreferenceValue(value) {
  const raw = String(value || '').trim().toLowerCase()
  if (!raw) return ''
  const slug = raw.replace(/[\s-]+/g, '_')
  const exact = DIET_PREFERENCE_OPTIONS.find((item) => item.value === slug)
  if (exact) return exact.value
  if (slug.includes('high') && slug.includes('protein')) return 'high_protein'
  if (slug.includes('low') && slug.includes('carb')) return 'low_carb'
  if (slug.includes('mediterranean')) return 'mediterranean'
  if (slug.includes('pescatarian')) return 'pescatarian'
  if (slug.startsWith('vegan')) return 'vegan'
  if (slug.includes('vegetarian')) return 'vegetarian'
  if (['balanced', 'balance', 'general', 'none', 'no_preference'].includes(slug)) return 'balanced'
  return slug
}

function formatDietPreferenceLabel(value) {
  const normalized = normalizeDietPreferenceValue(value)
  if (!normalized) return 'Set preference'
  const option = DIET_PREFERENCE_OPTIONS.find((item) => item.value === normalized)
  if (option) return option.label
  return String(value || '').replace(/_/g, ' ')
}

function formatActivityLabel(value) {
  const text = String(value || '').trim()
  if (!text) return ''
  return text.replace(/_/g, ' ')
}

function DietPreferenceSelect({ value, onChange, id = 'diet-preference-select' }) {
  const normalized = normalizeDietPreferenceValue(value)
  const hasLegacyValue = Boolean(value) && !DIET_PREFERENCE_OPTIONS.some((item) => item.value === normalized)
  return (
    <select
      id={id}
      className="input"
      value={hasLegacyValue ? value : normalized}
      onChange={(e) => onChange(e.target.value)}
    >
      {DIET_PREFERENCE_OPTIONS.map((option) => (
        <option key={option.value || 'none'} value={option.value}>{option.label}</option>
      ))}
      {hasLegacyValue ? <option value={value}>{String(value).replace(/_/g, ' ')} (current)</option> : null}
    </select>
  )
}

const ALLERGY_OPTIONS = [
  { value: 'shellfish', label: 'Shellfish' },
  { value: 'fish', label: 'Fish' },
  { value: 'peanuts', label: 'Peanuts' },
  { value: 'tree_nuts', label: 'Tree nuts' },
  { value: 'eggs', label: 'Eggs' },
  { value: 'milk', label: 'Milk / Dairy' },
  { value: 'soy', label: 'Soy' },
  { value: 'wheat', label: 'Wheat / Gluten' },
  { value: 'sesame', label: 'Sesame' },
]

function parseAllergyList(value) {
  if (Array.isArray(value)) return value.map((item) => String(item).trim()).filter(Boolean)
  return splitCsv(value)
}

function normalizeAllergyValue(value) {
  const slug = String(value || '').trim().toLowerCase().replace(/[\s-]+/g, '_')
  if (!slug) return ''
  const aliases = {
    shellfish: 'shellfish',
    fish: 'fish',
    peanut: 'peanut',
    peanuts: 'peanut',
    tree_nut: 'tree nut',
    tree_nuts: 'tree nut',
    egg: 'egg',
    eggs: 'egg',
    milk: 'milk',
    dairy: 'dairy',
    soy: 'soy',
    wheat: 'wheat',
    gluten: 'wheat',
    sesame: 'sesame',
  }
  return aliases[slug] || slug.replace(/_/g, ' ')
}

function normalizeAllergyList(values) {
  const normalized = []
  const seen = new Set()
  for (const item of parseAllergyList(values)) {
    const next = normalizeAllergyValue(item)
    if (!next || seen.has(next)) continue
    seen.add(next)
    normalized.push(next)
  }
  return normalized
}

function allergyListToCsv(values) {
  return normalizeAllergyList(values).join(', ')
}

function mergeAllergySelections(presetValue, otherValue) {
  return normalizeAllergyList([...parseAllergyList(presetValue), ...parseAllergyList(otherValue)])
}

function formatAllergyLabel(value) {
  const normalized = normalizeAllergyValue(value)
  const option = ALLERGY_OPTIONS.find((item) => normalizeAllergyValue(item.value) === normalized)
  if (option) return option.label
  return normalized.replace(/\b\w/g, (char) => char.toUpperCase())
}

function formatAllergiesLabel(allergies) {
  const list = normalizeAllergyList(allergies)
  if (!list.length) return 'None set'
  if (list.length === 1) return formatAllergyLabel(list[0])
  if (list.length === 2) return `${formatAllergyLabel(list[0])}, ${formatAllergyLabel(list[1])}`
  return `${formatAllergyLabel(list[0])} +${list.length - 1}`
}

function AllergyMultiSelect({ value, onChange, id = 'allergy-multi-select' }) {
  const selected = new Set(normalizeAllergyList(value))

  function toggle(optionValue) {
    const normalized = normalizeAllergyValue(optionValue)
    const next = new Set(selected)
    if (next.has(normalized)) next.delete(normalized)
    else next.add(normalized)
    onChange(allergyListToCsv([...next]))
  }

  return (
    <div className="allergy-multi-select" id={id} role="group" aria-label="Allergies">
      <div className="allergy-option-grid">
        {ALLERGY_OPTIONS.map((option) => {
          const normalized = normalizeAllergyValue(option.value)
          const active = selected.has(normalized)
          return (
            <button
              key={option.value}
              type="button"
              className={`allergy-option-chip${active ? ' active' : ''}`}
              aria-pressed={active}
              onClick={() => toggle(option.value)}
            >
              {option.label}
            </button>
          )
        })}
      </div>
    </div>
  )
}

function UserProfileFields({ value, onChange, includeWeight = false }) {
  const update = (field, next) => onChange({ ...value, [field]: next })
  return <>
    <input className="input" placeholder="Name" value={value.name} onChange={(e) => update('name', e.target.value)} />
    <select className="input" value={value.gender} onChange={(e) => update('gender', e.target.value)}><option value="male">Male</option><option value="female">Female</option></select>
    <input className="input" placeholder="Birth date (YYYYMMDD)" value={value.birth_date} onChange={(e) => update('birth_date', e.target.value)} />
    <input className="input" placeholder="Height (cm)" type="number" value={value.height_cm} onChange={(e) => update('height_cm', e.target.value)} />
    {includeWeight ? <input className="input" placeholder="Latest / initial weight (kg)" type="number" value={value.weight_kg} onChange={(e) => update('weight_kg', e.target.value)} /> : null}
    <input className="input" placeholder="Goal (e.g. lose_weight)" value={value.goal} onChange={(e) => update('goal', e.target.value)} />
    <input className="input" placeholder="Activity level" value={value.activity_level} onChange={(e) => update('activity_level', e.target.value)} />
    <DietPreferenceSelect value={value.diet_preference} onChange={(next) => update('diet_preference', next)} />
    <input className="input" placeholder="Budget level" value={value.budget_level} onChange={(e) => update('budget_level', e.target.value)} />
    <input className="input" placeholder="Target weight" value={value.target_weight} onChange={(e) => update('target_weight', e.target.value)} />
    <input className="input" placeholder="Target timeline" value={value.target_timeline} onChange={(e) => update('target_timeline', e.target.value)} />
    <input className="input" placeholder="Medical conditions (comma separated)" value={value.medical_conditions} onChange={(e) => update('medical_conditions', e.target.value)} />
    <AllergyMultiSelect value={value.allergies} onChange={(next) => update('allergies', next)} id="profile-allergies" />
    <input className="input" placeholder="Other allergies (comma separated, optional)" value={value.allergies_other || ''} onChange={(e) => update('allergies_other', e.target.value)} />
    <input className="input" placeholder="Food dislikes (comma separated)" value={value.food_dislikes} onChange={(e) => update('food_dislikes', e.target.value)} />
    <textarea className="input textarea-input" placeholder="Self description / routine / concerns" value={value.self_description} onChange={(e) => update('self_description', e.target.value)} />
    <textarea className="input textarea-input" placeholder="Coach notes" value={value.coach_notes} onChange={(e) => update('coach_notes', e.target.value)} />
  </>
}

function SummaryCard({ label, value }) { return <div className="summary-card"><div className="summary-label">{label}</div><div className="summary-value">{value}</div></div> }

function SettingsAboutModal({ health, onRefresh, onClose }) {
  const [tab, setTab] = useState('settings')
  const [refreshing, setRefreshing] = useState(false)

  async function handleRefresh() {
    setRefreshing(true)
    try {
      await onRefresh()
    } finally {
      setRefreshing(false)
    }
  }

  const ollamaOnline = Boolean(health?.ollama_reachable)
  const ragReady = Boolean(health?.rag_ready)

  return (
    <div className="modal-backdrop" onClick={onClose}>
      <div className="modal-card modal-card-wide settings-modal" onClick={(e) => e.stopPropagation()}>
        <div className="modal-head">
          <h3>Settings</h3>
          <button type="button" className="modal-close" onClick={onClose} aria-label="Close settings">×</button>
        </div>

        <div className="settings-tabs" role="tablist" aria-label="Settings sections">
          <button
            type="button"
            role="tab"
            aria-selected={tab === 'settings'}
            className={`window-chip ${tab === 'settings' ? 'active' : ''}`}
            onClick={() => setTab('settings')}
          >
            System
          </button>
          <button
            type="button"
            role="tab"
            aria-selected={tab === 'about'}
            className={`window-chip ${tab === 'about' ? 'active' : ''}`}
            onClick={() => setTab('about')}
          >
            About &amp; Help
          </button>
        </div>

        {tab === 'settings' ? (
          <div className="settings-panel" role="tabpanel">
            <p className="settings-lead">
              NutriCoachAI runs on your machine. Health profiles, chat history, and meal plans stay in local SQLite — not on a cloud coaching service.
            </p>

            <div className="settings-status-grid">
              <SettingsStatusRow
                label="Local AI engine (Ollama)"
                value={health ? (ollamaOnline ? 'Online' : 'Not running') : 'Checking…'}
                tone={health ? (ollamaOnline ? 'ok' : 'warn') : 'neutral'}
              />
              <SettingsStatusRow
                label="Chat model"
                value={health?.ollama_model || '—'}
                tone={ollamaOnline ? 'ok' : 'neutral'}
              />
              <SettingsStatusRow
                label="Summary model"
                value={health?.summary_model || '—'}
                tone="neutral"
              />
              <SettingsStatusRow
                label="Evidence base (RAG)"
                value={health ? (ragReady ? 'Ready' : 'Not indexed') : 'Checking…'}
                tone={health ? (ragReady ? 'ok' : 'warn') : 'neutral'}
              />
              <SettingsStatusRow
                label="Cross-session memory"
                value={health?.memory_mode ? `Mode ${health.memory_mode}` : health?.memory_feature_enabled ? 'Enabled' : '—'}
                tone="neutral"
              />
              <SettingsStatusRow
                label="Session idle timeout"
                value={health?.session_idle_timeout_minutes != null ? `${health.session_idle_timeout_minutes} min` : '—'}
                tone="neutral"
              />
            </div>

            {!ollamaOnline && health ? (
              <div className="settings-callout warn">
                Chat and meal-plan generation require Ollama. From the project root run <code>./start.sh</code> or start Ollama with <code>ollama serve</code>.
              </div>
            ) : null}

            {!ragReady && health ? (
              <div className="settings-callout subtle">
                RAG is unavailable when knowledge PDFs or embedding dependencies are missing. Coaching still works using profile and memory context.
              </div>
            ) : null}

            <div className="settings-actions">
              <button type="button" className="secondary-button" onClick={handleRefresh} disabled={refreshing}>
                {refreshing ? 'Refreshing…' : 'Refresh status'}
              </button>
            </div>
          </div>
        ) : (
          <div className="settings-panel" role="tabpanel">
            <section className="settings-section">
              <h4>About NutriCoachAI</h4>
              <p className="settings-about-text">
                NutriCoachAI is a <strong>privacy-preserving health coach chatbot</strong> built as an MSc research prototype.
                It combines local inference, retrieval-augmented generation (RAG) over trusted nutrition and activity guidelines,
                and compact cross-session memory so coaching can stay coherent without sending full chat transcripts to a cloud API.
              </p>
            </section>

            <section className="settings-section">
              <h4>Privacy-first architecture</h4>
              <ul className="settings-list">
                <li>User profiles, weights, chat, and meal plans are stored in a local SQLite database on this device.</li>
                <li>Coaching replies are generated by <strong>Ollama</strong> on <code>localhost</code> — the app does not call a cloud LLM API.</li>
                <li>Cross-session continuity uses summarized session notes injected into prompts, not unbounded transcript replay.</li>
                <li>
                  <strong>Residual network use:</strong> the first RAG run may download embedding model weights from Hugging Face Hub.
                  There is no user authentication; this build is intended for local demo and research use only.
                </li>
              </ul>
            </section>

            <section className="settings-section">
              <h4>Medical disclaimer</h4>
              <div className="settings-disclaimer-box" role="note">
                NutriCoachAI provides general wellness coaching only — not medical advice, diagnosis, or treatment.
                Always consult a qualified healthcare professional for personal medical concerns, medication changes, or emergency symptoms.
              </div>
            </section>

            <section className="settings-section">
              <h4>Getting started</h4>
              <ol className="settings-list ordered">
                <li>Install and start <strong>Ollama</strong>, then pull the configured chat model (default <code>deepseek-r1:8b</code>).</li>
                <li>From the project root, run <code>./start.sh</code> to launch the backend and frontend together.</li>
                <li>Open the app in a desktop or mobile browser, create a user profile, and start coaching.</li>
              </ol>
              <p className="settings-about-text muted">
                Stack: React + Vite frontend · FastAPI backend · SQLite · Chroma RAG · local Ollama (DeepSeek-R1).
              </p>
            </section>
          </div>
        )}
      </div>
    </div>
  )
}

function SettingsStatusRow({ label, value, tone = 'neutral' }) {
  return (
    <div className="settings-status-row">
      <div className="settings-status-label">{label}</div>
      <div className="settings-status-value">
        <span className={`status-badge ${tone}`}>{value}</span>
      </div>
    </div>
  )
}

const ONBOARDING_STEPS = [
  { id: 'welcome', title: 'Welcome' },
  { id: 'profile', title: 'About you' },
  { id: 'goal', title: 'Your goal' },
  { id: 'diet', title: 'Diet & allergies' },
  { id: 'weight', title: 'Starting point' },
]

const GOAL_OPTIONS = [
  { value: 'lose_weight', label: 'Lose weight' },
  { value: 'maintain_weight', label: 'Maintain weight' },
  { value: 'gain_muscle', label: 'Build muscle' },
  { value: 'eat_healthier', label: 'Eat healthier' },
  { value: 'general_wellness', label: 'General wellness' },
]

function OnboardingFlow({ preview, aiReady, error, onError, onComplete }) {
  const [stepIndex, setStepIndex] = useState(0)
  const [form, setForm] = useState(emptyOnboardingForm())
  const [submitting, setSubmitting] = useState(false)
  const [stepError, setStepError] = useState('')

  const step = ONBOARDING_STEPS[stepIndex]
  const progressSteps = ONBOARDING_STEPS.filter((item) => item.id !== 'welcome')
  const progressIndex = Math.max(0, stepIndex - 1)

  function updateField(field, value) {
    setForm((prev) => ({ ...prev, [field]: value }))
    setStepError('')
  }

  function validateCurrentStep() {
    if (step.id === 'welcome') return ''
    if (step.id === 'profile') {
      if (!form.name.trim()) return 'Please enter your name.'
      if (!/^\d{8}$/.test(form.birth_date.trim())) return 'Birth date must be 8 digits (YYYYMMDD).'
      if (!form.height_cm || Number(form.height_cm) <= 0) return 'Please enter a valid height in cm.'
      return ''
    }
    if (step.id === 'goal') {
      if (!form.goal) return 'Please choose a health goal.'
      return ''
    }
    if (step.id === 'diet') return ''
    if (step.id === 'weight') {
      if (!form.weight_kg || Number(form.weight_kg) <= 0) return 'Please enter your current weight in kg.'
      return ''
    }
    return ''
  }

  function goNext() {
    const message = validateCurrentStep()
    if (message) {
      setStepError(message)
      return
    }
    setStepError('')
    setStepIndex((index) => Math.min(index + 1, ONBOARDING_STEPS.length - 1))
  }

  function goBack() {
    setStepError('')
    setStepIndex((index) => Math.max(index - 1, 0))
  }

  async function handleSubmit(e) {
    e.preventDefault()
    const message = validateCurrentStep()
    if (message) {
      setStepError(message)
      return
    }
    setSubmitting(true)
    onError('')
    try {
      await onComplete(form)
    } catch (err) {
      onError(err.message)
    } finally {
      setSubmitting(false)
    }
  }

  return (
    <div className="onboarding-screen">
      <div className="onboarding-shell">
        <header className="onboarding-header">
          <div className="onboarding-brand">
            <div className="onboarding-brand-mark">N</div>
            <div>
              <div className="onboarding-brand-title">NutriCoachAI</div>
              <div className="onboarding-brand-subtitle">Local-first health coaching</div>
            </div>
          </div>
          {preview ? <span className="onboarding-preview-badge">Preview</span> : null}
        </header>

        {step.id !== 'welcome' ? (
          <div className="onboarding-progress" aria-label="Onboarding progress">
            {progressSteps.map((item, index) => (
              <div
                key={item.id}
                className={`onboarding-progress-dot ${index <= progressIndex ? 'active' : ''} ${index === progressIndex ? 'current' : ''}`}
              />
            ))}
            <div className="onboarding-progress-label">
              Step {progressIndex + 1} of {progressSteps.length} · {step.title}
            </div>
          </div>
        ) : null}

        <div className="onboarding-card">
          {step.id === 'welcome' ? (
            <>
              <h1 className="onboarding-title">Welcome to your private health coach</h1>
              <p className="onboarding-lead">
                NutriCoachAI runs on your device. Your profile, chat, and progress stay in local storage — not on a cloud coaching account.
              </p>
              <ul className="onboarding-feature-list">
                <li>Personalized nutrition and wellness guidance</li>
                <li>Evidence-grounded answers from trusted guidelines</li>
                <li>Coach memory that carries across conversations</li>
              </ul>
              {!aiReady ? (
                <div className="onboarding-callout warn">
                  Local AI (Ollama) is not running yet. You can finish setup now; start Ollama before chatting.
                </div>
              ) : null}
              <button type="button" className="cta-button onboarding-primary" onClick={goNext}>
                Get started
              </button>
            </>
          ) : null}

          {step.id === 'profile' ? (
            <>
              <h2 className="onboarding-title">Tell us about you</h2>
              <p className="onboarding-lead">Basic details help calculate BMI and estimated REE, and tailor coaching.</p>
              <div className="onboarding-form">
                <input className="input" placeholder="Your name" value={form.name} onChange={(e) => updateField('name', e.target.value)} />
                <select className="input" value={form.gender} onChange={(e) => updateField('gender', e.target.value)}>
                  <option value="male">Male</option>
                  <option value="female">Female</option>
                </select>
                <input className="input" placeholder="Birth date (YYYYMMDD)" value={form.birth_date} onChange={(e) => updateField('birth_date', e.target.value)} />
                <input className="input" type="number" placeholder="Height (cm)" value={form.height_cm} onChange={(e) => updateField('height_cm', e.target.value)} />
              </div>
            </>
          ) : null}

          {step.id === 'goal' ? (
            <>
              <h2 className="onboarding-title">What is your health goal?</h2>
              <p className="onboarding-lead">Your coach will use this to keep advice focused and realistic.</p>
              <div className="onboarding-form">
                <select className="input" value={form.goal} onChange={(e) => updateField('goal', e.target.value)}>
                  <option value="">Select a goal</option>
                  {GOAL_OPTIONS.map((option) => (
                    <option key={option.value} value={option.value}>{option.label}</option>
                  ))}
                </select>
                <input className="input" placeholder="Target weight (optional, e.g. 59 kg)" value={form.target_weight} onChange={(e) => updateField('target_weight', e.target.value)} />
                <input className="input" placeholder="Target timeline (optional, e.g. 3 months)" value={form.target_timeline} onChange={(e) => updateField('target_timeline', e.target.value)} />
                <select className="input" value={form.activity_level} onChange={(e) => updateField('activity_level', e.target.value)}>
                  <option value="">Activity level (optional)</option>
                  <option value="sedentary">Mostly sedentary</option>
                  <option value="lightly_active">Lightly active</option>
                  <option value="moderately_active">Moderately active</option>
                  <option value="very_active">Very active</option>
                </select>
              </div>
            </>
          ) : null}

          {step.id === 'diet' ? (
            <>
              <h2 className="onboarding-title">Diet preferences &amp; allergies</h2>
              <p className="onboarding-lead">These constraints are always included in your coach context.</p>
              <div className="onboarding-form">
                <DietPreferenceSelect value={form.diet_preference} onChange={(next) => updateField('diet_preference', next)} id="onboarding-diet-preference" />
                <AllergyMultiSelect value={form.allergies} onChange={(next) => updateField('allergies', next)} id="onboarding-allergies" />
                <input className="input" placeholder="Other allergies (comma separated, optional)" value={form.allergies_other} onChange={(e) => updateField('allergies_other', e.target.value)} />
                <input className="input" placeholder="Foods to avoid (comma separated, optional)" value={form.food_dislikes} onChange={(e) => updateField('food_dislikes', e.target.value)} />
              </div>
            </>
          ) : null}

          {step.id === 'weight' ? (
            <>
              <h2 className="onboarding-title">Your starting weight</h2>
              <p className="onboarding-lead">We will track progress from this baseline and show BMI and estimated REE on your dashboard.</p>
              <form className="onboarding-form" onSubmit={handleSubmit}>
                <input className="input" type="number" step="0.1" placeholder="Current weight (kg)" value={form.weight_kg} onChange={(e) => updateField('weight_kg', e.target.value)} />
                <div className="onboarding-review">
                  <div className="onboarding-review-title">Ready to coach {form.name || 'you'}</div>
                  <div className="onboarding-review-grid">
                    <span>Goal: {GOAL_OPTIONS.find((item) => item.value === form.goal)?.label || form.goal}</span>
                    {form.diet_preference ? <span>Diet: {formatDietPreferenceLabel(form.diet_preference)}</span> : null}
                    {form.allergies || form.allergies_other ? <span>Allergies: {formatAllergiesLabel(mergeAllergySelections(form.allergies, form.allergies_other))}</span> : null}
                    {form.height_cm ? <span>Height: {form.height_cm} cm</span> : null}
                  </div>
                </div>
                <button type="submit" className="cta-button onboarding-primary" disabled={submitting}>
                  {submitting ? 'Creating profile…' : 'Start coaching'}
                </button>
              </form>
            </>
          ) : null}

          {stepError ? <div className="onboarding-step-error">{stepError}</div> : null}
          {error ? <div className="onboarding-step-error">{error}</div> : null}

          {step.id !== 'welcome' && step.id !== 'weight' ? (
            <div className="onboarding-actions">
              <button type="button" className="secondary-button" onClick={goBack}>Back</button>
              <button type="button" className="cta-button" onClick={goNext}>Continue</button>
            </div>
          ) : null}

          {step.id === 'welcome' ? null : step.id !== 'weight' ? null : (
            <div className="onboarding-actions single">
              <button type="button" className="secondary-button" onClick={goBack}>Back</button>
            </div>
          )}
        </div>

        <p className="onboarding-footnote">
          General wellness coaching only — not medical advice. Consult a healthcare professional for medical concerns.
        </p>
      </div>
    </div>
  )
}

function MemoryViewerModal({ userId, user, metrics, aiReady, onClose, onMemoryUpdated }) {
  const [loading, setLoading] = useState(true)
  const [loadError, setLoadError] = useState('')
  const [memory, setMemory] = useState(null)
  const [summaries, setSummaries] = useState([])
  const [sessions, setSessions] = useState([])
  const [expandedSessionId, setExpandedSessionId] = useState(null)
  const [regeneratingSessionId, setRegeneratingSessionId] = useState(null)
  const [actionError, setActionError] = useState('')

  useEffect(() => {
    loadMemoryData()
  }, [userId])

  async function loadMemoryData() {
    setLoading(true)
    setLoadError('')
    try {
      const [mem, sum, sess] = await Promise.all([
        api(`/users/${userId}/memory`),
        api(`/users/${userId}/summaries?limit=20`),
        api(`/users/${userId}/sessions?limit=20`),
      ])
      setMemory(mem)
      setSummaries(sum)
      setSessions(sess)
      onMemoryUpdated?.(mem)
    } catch (err) {
      setLoadError(err.message)
    } finally {
      setLoading(false)
    }
  }

  async function handleRegenerate(sessionId) {
    setRegeneratingSessionId(sessionId)
    setActionError('')
    try {
      await api(`/users/${userId}/summaries/regenerate`, {
        method: 'POST',
        body: JSON.stringify({ session_id: sessionId }),
      })
      await loadMemoryData()
      setExpandedSessionId(sessionId)
    } catch (err) {
      setActionError(err.message)
    } finally {
      setRegeneratingSessionId(null)
    }
  }

  const profileFacts = buildProfileMemoryFacts(user, metrics)
  const sessionSummaries = summaries.filter((item) => item.summary_type === 'session' && !item.archived)
  const archivedCount = summaries.filter((item) => item.summary_type === 'session' && item.archived).length
  const summaryBySessionId = new Map(
    summaries
      .filter((item) => item.summary_type === 'session' && item.session_id != null)
      .map((item) => [item.session_id, item])
  )

  return (
    <div className="modal-backdrop" onClick={onClose}>
      <div className="modal-card modal-card-wide memory-viewer-modal" onClick={(e) => e.stopPropagation()}>
        <div className="modal-head">
          <div>
            <h3>Coach Memory</h3>
            <p className="memory-viewer-subtitle">What NutriCoachAI remembers for {user?.name || 'this user'}</p>
          </div>
          <button type="button" className="modal-close" onClick={onClose} aria-label="Close memory viewer">×</button>
        </div>

        <p className="memory-viewer-lead">
          Cross-session coaching uses compact summaries — not full chat transcripts — plus profile and metrics injected into each reply (memory mode M2).
        </p>

        {loadError ? <div className="memory-viewer-error">{loadError}</div> : null}
        {actionError ? <div className="memory-viewer-error">{actionError}</div> : null}

        {loading ? (
          <div className="memory-viewer-loading">Loading memory…</div>
        ) : (
          <div className="memory-viewer-body">
            <section className="memory-section">
              <h4>Profile context</h4>
              <p className="memory-section-note">Always included in the coach prompt alongside latest metrics.</p>
              {profileFacts.length ? (
                <div className="memory-facts-grid">
                  {profileFacts.map((fact) => (
                    <div className="memory-fact-card" key={fact.label}>
                      <div className="memory-fact-label">{fact.label}</div>
                      <div className="memory-fact-value">{fact.value}</div>
                    </div>
                  ))}
                </div>
              ) : (
                <div className="memory-empty">No profile coaching fields saved yet. Edit the user profile to add goals and constraints.</div>
              )}
            </section>

            <section className="memory-section">
              <h4>Long-term cumulative memory</h4>
              <p className="memory-section-note">
                Rolled-up coaching memory merged from older session summaries.
                {memory?.last_rollup_at ? ` Last updated ${formatMemoryTimestamp(memory.last_rollup_at)}.` : ''}
              </p>
              {memory?.cumulative_summary ? (
                <div className="memory-block">
                  <pre className="memory-block-content">{memory.cumulative_summary}</pre>
                </div>
              ) : (
                <div className="memory-empty">
                  No long-term memory yet. Long-term cumulative memory appears after you close{' '}
                  {ROLLUP_SESSION_THRESHOLD} or more conversations (each with at least 2 messages); older
                  session summaries then roll up into this block.
                  {sessionSummaries.length > 0
                    ? ` You currently have ${sessionSummaries.length} recent session summar${sessionSummaries.length === 1 ? 'y' : 'ies'}.`
                    : ' Close a conversation with New Conversation after chatting to create your first session summary.'}
                </div>
              )}
              {archivedCount > 0 ? (
                <p className="memory-section-note muted">{archivedCount} older session summar{archivedCount === 1 ? 'y was' : 'ies were'} merged into long-term memory.</p>
              ) : null}
            </section>

            <section className="memory-section">
              <h4>Recent session summaries</h4>
              <p className="memory-section-note">Injected into new chats after you close a conversation (New Conversation).</p>
              {sessionSummaries.length ? (
                <div className="memory-summary-list">
                  {sessionSummaries.map((item) => (
                    <div className="memory-summary-card" key={item.summary_id}>
                      <div className="memory-summary-head">
                        <span className="memory-summary-title">
                          Session #{item.session_id ?? '—'}
                        </span>
                        <span className="memory-summary-meta">
                          {formatMemoryTimestamp(item.created_at)}
                          {item.message_count != null ? ` · ${item.message_count} messages` : ''}
                        </span>
                      </div>
                      <pre className="memory-block-content compact">{item.content}</pre>
                    </div>
                  ))}
                </div>
              ) : (
                <div className="memory-empty">
                  No session summaries yet. Chat for at least two turns, then use New Conversation to create one.
                </div>
              )}
            </section>

            <section className="memory-section">
              <div className="memory-section-head">
                <h4>Session history</h4>
                <button type="button" className="secondary-button memory-refresh-button" onClick={loadMemoryData}>
                  Refresh
                </button>
              </div>
              {sessions.length ? (
                <div className="memory-session-list">
                  {sessions.map((session) => {
                    const fullSummary = summaryBySessionId.get(session.session_id)
                    const expanded = expandedSessionId === session.session_id
                    const canRegenerate = session.status === 'closed' && (session.turn_count ?? 0) >= 2
                    return (
                      <div className="memory-session-card" key={session.session_id}>
                        <div className="memory-session-top">
                          <div>
                            <div className="memory-session-title">
                              Session #{session.session_id}
                              <span className={`status-badge ${session.status === 'active' ? 'ok' : 'neutral'}`}>
                                {session.status}
                              </span>
                            </div>
                            <div className="memory-session-meta">
                              Started {formatMemoryTimestamp(session.started_at)}
                              {session.ended_at ? ` · Ended ${formatMemoryTimestamp(session.ended_at)}` : ''}
                              {session.turn_count != null ? ` · ${session.turn_count} user turns` : ''}
                            </div>
                          </div>
                          <div className="memory-session-actions">
                            {fullSummary?.content || session.summary_preview ? (
                              <button
                                type="button"
                                className="table-button"
                                onClick={() => setExpandedSessionId(expanded ? null : session.session_id)}
                              >
                                {expanded ? 'Hide summary' : 'View summary'}
                              </button>
                            ) : null}
                            {canRegenerate ? (
                              <button
                                type="button"
                                className="table-button"
                                disabled={!aiReady || regeneratingSessionId === session.session_id}
                                title={!aiReady ? 'Start Ollama to regenerate summaries' : undefined}
                                onClick={() => handleRegenerate(session.session_id)}
                              >
                                {regeneratingSessionId === session.session_id ? 'Regenerating…' : 'Regenerate'}
                              </button>
                            ) : null}
                          </div>
                        </div>
                        {!expanded && session.summary_preview ? (
                          <p className="memory-session-preview">{session.summary_preview}</p>
                        ) : null}
                        {expanded && (fullSummary?.content || session.summary_preview) ? (
                          <pre className="memory-block-content compact">{fullSummary?.content || session.summary_preview}</pre>
                        ) : null}
                        {!fullSummary?.content && !session.summary_preview && session.status === 'closed' ? (
                          <p className="memory-session-preview muted">
                            {canRegenerate ? 'Summary pending or not yet generated.' : 'Too few messages to summarize.'}
                          </p>
                        ) : null}
                      </div>
                    )
                  })}
                </div>
              ) : (
                <div className="memory-empty">No chat sessions recorded yet.</div>
              )}
            </section>

            {memory?.active_session ? (
              <section className="memory-section">
                <h4>Active session</h4>
                <p className="memory-section-note">
                  Session #{memory.active_session.session_id} is open — recent turns from this session are also injected for short-term coherence.
                </p>
              </section>
            ) : null}
          </div>
        )}
      </div>
    </div>
  )
}

function buildProfileMemoryFacts(user, metrics) {
  if (!user) return []
  const facts = []
  const add = (label, value) => {
    const text = value == null ? '' : String(value).trim()
    if (text) facts.push({ label, value: text })
  }

  add('Goal', user.goal?.replace(/_/g, ' '))
  add('Target weight', user.target_weight)
  add('Target timeline', user.target_timeline)
  add('Activity level', user.activity_level)
  if (user.diet_preference) add('Diet preference', formatDietPreferenceLabel(user.diet_preference))
  add('Budget', user.budget_level)
  if (metrics?.weight_kg != null) add('Current weight', `${metrics.weight_kg} kg`)
  if (metrics?.bmi != null) add('BMI', metrics.bmi_label ? `${metrics.bmi} (${metrics.bmi_label})` : String(metrics.bmi))
  if (metrics?.ree != null) add('Estimated REE', `${metrics.ree} kcal/day`)
  if (Array.isArray(user.allergies) && user.allergies.length) add('Allergies', user.allergies.map(formatAllergyLabel).join(', '))
  if (Array.isArray(user.medical_conditions) && user.medical_conditions.length) add('Medical conditions', user.medical_conditions.join(', '))
  if (Array.isArray(user.food_dislikes) && user.food_dislikes.length) add('Food dislikes', user.food_dislikes.join(', '))
  add('Self description', user.self_description)
  add('Coach notes', user.coach_notes)
  return facts
}

function formatMemoryTimestamp(raw) {
  if (!raw) return '—'
  const date = parseRecordDate(raw)
  if (Number.isNaN(date.getTime())) return String(raw)
  return new Intl.DateTimeFormat('en-GB', {
    day: 'numeric',
    month: 'short',
    year: 'numeric',
    hour: '2-digit',
    minute: '2-digit',
  }).format(date)
}

function Modal({ title, children, onClose }) { return <div className="modal-backdrop" onClick={onClose}><div className="modal-card" onClick={(e) => e.stopPropagation()}><div className="modal-head"><h3>{title}</h3><button className="modal-close" onClick={onClose}>×</button></div>{children}</div></div> }
function ConfirmModal({ title, body, confirmLabel, onCancel, onConfirm }) { return <div className="modal-backdrop" onClick={onCancel}><div className="modal-card" onClick={(e) => e.stopPropagation()}><div className="modal-head"><h3>{title}</h3><button className="modal-close" onClick={onCancel}>×</button></div><p className="confirm-text">{body}</p><div className="confirm-actions"><button className="secondary-button" onClick={onCancel}>Cancel</button><button className="danger-button confirm" onClick={onConfirm}>{confirmLabel}</button></div></div></div> }

function normalizeSeries(series) {
  return [...series].map((item) => ({ ...item, weight_kg: Number(item.weight_kg), bmi: Number(item.bmi), ree: Number(item.ree), date: parseRecordDate(item.record_date || item.recorded_at), dateKey: item.record_date || item.recorded_at })).filter((item) => Number.isFinite(item.weight_kg) && item.date instanceof Date && !Number.isNaN(item.date.getTime())).sort((a, b) => a.date - b.date)
}
function filterSeriesByDays(series, days) {
  if (!series.length) return []
  const latest = series.at(-1).date
  const start = new Date(latest)
  start.setDate(start.getDate() - (days - 1))
  const filtered = series.filter((item) => item.date >= start && item.date <= latest)
  return filtered.length ? filtered : series.slice(-Math.min(series.length, 5))
}
function buildSidebarChart(series) {
  const chart = buildChart(series, { width: 244, height: 152, padTop: 16, padBottom: 36, padLeft: 42, padRight: 12, xTickMode: 'compact', showPointDots: true })
  if (!series.length) return chart
  const values = series.map((item) => item.weight_kg)
  const min = Math.min(...values)
  const max = Math.max(...values)
  const mid = Number(((min + max) / 2).toFixed(1))
  let sidebarTicks = [min, mid, max]
  sidebarTicks = Array.from(new Set(sidebarTicks.map((v) => Number(v.toFixed(1))))).sort((a, b) => a - b)
  chart.yTicks = sidebarTicks.map((value) => {
    const y = chart.area.bottom - ((value - chart.minY) / chart.yRange) * (chart.area.bottom - chart.area.top)
    return { value, y }
  })
  if (series.length <= 2) {
    chart.xTicks = series.map((item) => ({ label: formatShortDate(item.date), x: scaleDate(item.date, series, chart.area) }))
  } else {
    chart.xTicks = [series[0], series.at(-1)].map((item) => ({ label: formatShortDate(item.date), x: scaleDate(item.date, series, chart.area) }))
  }
  return chart
}

function buildChart(series, opts) {
  const { width, height, padTop, padBottom, padLeft, padRight, xTickMode = 'range', showPointDots = true } = opts
  const area = { left: padLeft, right: width - padRight, top: padTop, bottom: height - padBottom }
  const innerW = area.right - area.left
  const innerH = area.bottom - area.top
  if (!series.length) return { area, yTicks: [], xTicks: [], points: [], path: '', latestPoint: null }
  const weights = series.map((item) => item.weight_kg)
  const minW = Math.min(...weights)
  const maxW = Math.max(...weights)
  const step = pickNiceStep(Math.max(1, maxW - minW))
  const minY = Math.floor((minW - step) / step) * step
  const maxY = Math.ceil((maxW + step) / step) * step
  const yRange = Math.max(step, maxY - minY)
  const yTicks = []
  for (let value = minY; value <= maxY + 0.0001; value += step) yTicks.push({ value: cleanNumber(value), y: area.bottom - ((value - minY) / yRange) * innerH })
  const startTime = series[0].date.getTime()
  const endTime = series.at(-1).date.getTime()
  const timeRange = Math.max(1, endTime - startTime)
  const points = series.map((item) => ({ ...item, x: area.left + ((item.date.getTime() - startTime) / timeRange) * innerW, y: area.bottom - ((item.weight_kg - minY) / yRange) * innerH }))
  const path = points.map((point, index) => `${index === 0 ? 'M' : 'L'} ${point.x.toFixed(2)} ${point.y.toFixed(2)}`).join(' ')
  return { area, yTicks, xTicks: buildDateTicks(series, area, xTickMode), points: showPointDots ? points : [], path, latestPoint: points.at(-1) || null, minY, maxY, yRange }
}
function buildDateTicks(series, area, mode = 'range') {
  if (!series.length) return []
  const byIndex = (indexes) => Array.from(new Set(indexes)).map((i) => series[i]).filter(Boolean).map((item) => ({ label: formatShortDate(item.date), x: scaleDate(item.date, series, area) }))
  if (mode === 'compact') {
    if (series.length <= 2) return byIndex(series.map((_, i) => i))
    return byIndex([0, series.length - 1])
  }
  if (series.length <= 4) return byIndex(series.map((_, i) => i))
  return byIndex([0, Math.floor((series.length - 1) / 3), Math.floor(((series.length - 1) * 2) / 3), series.length - 1])
}
function scaleDate(date, series, area) { const start = series[0].date.getTime(); const end = series.at(-1).date.getTime(); const range = Math.max(1, end - start); return area.left + ((date.getTime() - start) / range) * (area.right - area.left) }
function pickNiceStep(range) { if (range <= 6) return 1; if (range <= 14) return 2; if (range <= 30) return 5; return 10 }
function parseRecordDate(raw) { const text = String(raw ?? ''); if (/^\d{8}$/.test(text)) return new Date(Number(text.slice(0,4)), Number(text.slice(4,6))-1, Number(text.slice(6,8))); return new Date(text) }
function formatShortDate(date) { return new Intl.DateTimeFormat('en-GB', { day: 'numeric', month: 'short' }).format(date) }
function formatLongDate(date) { return new Intl.DateTimeFormat('en-GB', { day: 'numeric', month: 'short', year: 'numeric' }).format(date) }
function cleanNumber(value) { return Number.isInteger(value) ? value : Number(value.toFixed(1)) }
function toDateTimeLocal(date) { const d = date instanceof Date ? date : new Date(date); if (Number.isNaN(d.getTime())) return ''; const pad = (v) => String(v).padStart(2, '0'); return `${d.getFullYear()}-${pad(d.getMonth()+1)}-${pad(d.getDate())}T${pad(d.getHours())}:${pad(d.getMinutes())}` }
function fromDateTimeLocal(value) { if (!value) return null; return value.replace('T', ' ') + ':00' }


function emptyUserForm() {
  return {
    name: '', gender: 'male', birth_date: '', height_cm: '', weight_kg: '',
    goal: '', activity_level: '', diet_preference: '', budget_level: '', target_weight: '', target_timeline: '',
    medical_conditions: '', allergies: '', allergies_other: '', food_dislikes: '', self_description: '', coach_notes: ''
  }
}

function emptyOnboardingForm() {
  return {
    ...emptyUserForm(),
    goal: 'lose_weight',
  }
}

function readOnboardingPreviewFlag() {
  if (typeof window === 'undefined') return false
  return new URLSearchParams(window.location.search).get('onboarding') === 'preview'
}

function parseTargetWeightKg(value) {
  const match = String(value || '').match(/(\d+(?:\.\d+)?)/)
  if (!match) return null
  const parsed = Number(match[1])
  return Number.isFinite(parsed) && parsed > 0 ? parsed : null
}

function formatGoalLabel(goal) {
  const text = String(goal || '').trim()
  if (!text) return 'Your coaching goal'
  return text.replace(/_/g, ' ').replace(/\b\w/g, (char) => char.toUpperCase())
}

function inferGoalDirection(goal) {
  const normalized = String(goal || '').toLowerCase()
  if (normalized.includes('lose') || normalized.includes('cut')) return 'lose'
  if (normalized.includes('gain') || normalized.includes('muscle') || normalized.includes('bulk')) return 'gain'
  return 'neutral'
}

function clampPercent(value) {
  if (!Number.isFinite(value)) return 0
  return Math.max(0, Math.min(100, Math.round(value)))
}

function formatKg(value) {
  if (!Number.isFinite(value)) return '--'
  return `${Number(value.toFixed(1))} kg`
}

function buildGoalProgress(user, metrics, weightSeries) {
  const targetKg = parseTargetWeightKg(user?.target_weight)
  const currentKg = Number(metrics?.weight_kg)
  if (!targetKg || !Number.isFinite(currentKg)) return null

  const series = weightSeries?.length ? weightSeries : [{ weight_kg: currentKg }]
  const startKg = Number(series[0]?.weight_kg ?? currentKg)
  const direction = inferGoalDirection(user?.goal)

  let progressPercent = 0
  let remainingKg = Math.abs(currentKg - targetKg)
  let remainingLabel = 'Remaining'

  if (direction === 'lose' || (direction === 'neutral' && targetKg < startKg)) {
    const total = startKg - targetKg
    const completed = startKg - currentKg
    progressPercent = total > 0 ? clampPercent((completed / total) * 100) : (currentKg <= targetKg ? 100 : 0)
    remainingKg = Math.max(0, currentKg - targetKg)
    remainingLabel = 'To target'
  } else if (direction === 'gain' || (direction === 'neutral' && targetKg > startKg)) {
    const total = targetKg - startKg
    const completed = currentKg - startKg
    progressPercent = total > 0 ? clampPercent((completed / total) * 100) : (currentKg >= targetKg ? 100 : 0)
    remainingKg = Math.max(0, targetKg - currentKg)
    remainingLabel = 'To target'
  } else {
    const span = Math.abs(startKg - targetKg)
    const traveled = Math.abs(currentKg - startKg)
    progressPercent = span > 0 ? clampPercent((traveled / span) * 100) : 0
    remainingKg = Math.abs(currentKg - targetKg)
    remainingLabel = 'From target'
  }

  return {
    currentKg,
    targetKg,
    startKg,
    remainingKg,
    remainingLabel,
    progressPercent,
    goalLabel: formatGoalLabel(user?.goal),
    targetTimeline: String(user?.target_timeline || '').trim(),
    trendMessage: buildWeightTrendMessage(series, direction),
  }
}

function buildWeightTrendMessage(series, direction) {
  if (!series?.length || series.length < 2) {
    return 'Log more weigh-ins to unlock a recent trend message from your weight history.'
  }

  const latest = Number(series[series.length - 1]?.weight_kg)
  const previous = Number(series[series.length - 2]?.weight_kg)
  if (!Number.isFinite(latest) || !Number.isFinite(previous)) {
    return 'Keep logging weight to track your recent trend.'
  }

  const delta = latest - previous
  const absDelta = Math.abs(delta).toFixed(1)

  if (Math.abs(delta) < 0.1) {
    return 'Your weight is steady compared with your previous weigh-in.'
  }

  if (series.length >= 3) {
    const first = Number(series[0]?.weight_kg)
    const spanDelta = latest - first
    const spanAbs = Math.abs(spanDelta).toFixed(1)
    if (Math.abs(spanDelta) >= 0.1) {
      const spanDirection = spanDelta < 0 ? 'down' : 'up'
      const recentDirection = delta < 0 ? 'down' : 'up'
      const aligned = (direction === 'lose' && spanDelta < 0) || (direction === 'gain' && spanDelta > 0)
      if (aligned) {
        return `Trending ${spanDirection} ${spanAbs} kg since your first logged weigh-in, including ${recentDirection} ${absDelta} kg since the last entry.`
      }
      return `Overall ${spanDirection} ${spanAbs} kg since your first logged weigh-in. Latest change: ${recentDirection} ${absDelta} kg.`
    }
  }

  if (delta < 0) {
    return direction === 'lose'
      ? `Down ${absDelta} kg since your previous weigh-in — moving toward your target.`
      : `Down ${absDelta} kg since your previous weigh-in.`
  }

  return direction === 'gain'
    ? `Up ${absDelta} kg since your previous weigh-in — moving toward your target.`
    : `Up ${absDelta} kg since your previous weigh-in.`
}

function userToForm(user, metrics) {
  const allAllergies = normalizeAllergyList(user?.allergies)
  const known = new Set(ALLERGY_OPTIONS.map((item) => normalizeAllergyValue(item.value)))
  const presetAllergies = allAllergies.filter((item) => known.has(item))
  const customAllergies = allAllergies.filter((item) => !known.has(item))
  return {
    name: user?.name || '',
    gender: user?.gender || 'male',
    birth_date: user?.birth_date || '',
    height_cm: user?.height_cm || '',
    weight_kg: metrics?.weight_kg || '',
    goal: user?.goal || '',
    activity_level: user?.activity_level || '',
    diet_preference: user?.diet_preference || '',
    budget_level: user?.budget_level || '',
    target_weight: user?.target_weight || '',
    target_timeline: user?.target_timeline || '',
    medical_conditions: Array.isArray(user?.medical_conditions) ? user.medical_conditions.join(', ') : '',
    allergies: allergyListToCsv(presetAllergies),
    allergies_other: customAllergies.join(', '),
    food_dislikes: Array.isArray(user?.food_dislikes) ? user.food_dislikes.join(', ') : '',
    self_description: user?.self_description || '',
    coach_notes: user?.coach_notes || '',
  }
}

function toUserPayload(form) {
  const { allergies_other, ...rest } = form
  return {
    ...rest,
    height_cm: Number(form.height_cm),
    weight_kg: form.weight_kg ? Number(form.weight_kg) : null,
    medical_conditions: splitCsv(form.medical_conditions),
    allergies: mergeAllergySelections(form.allergies, allergies_other),
    food_dislikes: splitCsv(form.food_dislikes),
  }
}

function splitCsv(value) {
  return String(value || '').split(',').map((item) => item.trim()).filter(Boolean)
}

function isFoodChoiceUsable(comparison) {
  if (!comparison || typeof comparison !== 'object') return false
  const optionA = String(comparison.option_a || '').trim()
  const optionB = String(comparison.option_b || '').trim()
  if (!optionA || !optionB) return false
  if (
    ['option a', 'option_a'].includes(optionA.toLowerCase()) &&
    ['option b', 'option_b'].includes(optionB.toLowerCase())
  ) {
    return false
  }
  const dims = comparison.comparison || {}
  const required = ['protein', 'carbs', 'sodium', 'glycemic']
  const filled = required.filter((key) => {
    const row = dims[key] || {}
    return String(row.option_a || '').trim() && String(row.option_b || '').trim()
  }).length
  const hasRecommendation = Boolean(String(comparison.recommendation || '').trim())
  return filled >= required.length && hasRecommendation
}

function parseFoodChoiceFromContent(content) {
  const marker = '<!--nutricoach-food-choice'
  const text = String(content || '')
  const start = text.indexOf(marker)
  if (start === -1) return { visible: text, foodChoice: null }
  const end = text.indexOf('-->', start)
  if (end === -1) return { visible: text, foodChoice: null }
  const payloadRaw = text.slice(start + marker.length, end).trim()
  const visible = text.slice(0, start).trim()
  try {
    const parsed = JSON.parse(payloadRaw)
    if (!parsed || typeof parsed !== 'object') return { visible: visible || text, foodChoice: null }
    return { visible: visible || text, foodChoice: isFoodChoiceUsable(parsed) ? parsed : null }
  } catch {
    return { visible: visible || text, foodChoice: null }
  }
}

function normalizeChatHistory(payload) {
  const messages = Array.isArray(payload) ? payload : (payload && Array.isArray(payload.messages) ? payload.messages : [])
  return messages.map((item) => {
    if (!item || item.role === 'user' || item.foodChoice) return item
    const { visible, foodChoice } = parseFoodChoiceFromContent(item.content)
    return {
      ...item,
      content: visible,
      ...(foodChoice ? { foodChoice } : {}),
    }
  })
}

function isMobileViewport() {
  if (typeof window === 'undefined') return false
  return window.matchMedia('(max-width: 960px)').matches
}

async function api(path, options = {}) {
  const response = await fetch(`${API_BASE}${path}`, { headers: { 'Content-Type': 'application/json', ...(options.headers || {}) }, ...options })
  if (!response.ok) {
    let detail = 'Request failed'
    try { const data = await response.json(); detail = data.detail || detail } catch { detail = `${response.status} ${response.statusText}` }
    throw new Error(detail)
  }
  return response.json()
}

function dispatchSSEBlock(block, handlers) {
  let eventName = 'message'
  const dataLines = []
  for (const line of block.split('\n')) {
    if (!line || line.startsWith(':')) continue
    if (line.startsWith('event:')) eventName = line.slice(6).trim()
    else if (line.startsWith('data:')) dataLines.push(line.slice(5).trim())
  }
  if (!dataLines.length) return
  const data = JSON.parse(dataLines.join('\n'))
  if (eventName === 'meta' && handlers.onMeta) handlers.onMeta(data)
  if (eventName === 'token' && handlers.onToken) handlers.onToken(data.text || '')
  if (eventName === 'done' && handlers.onDone) handlers.onDone(data)
  if (eventName === 'error') throw new Error(data.detail || 'Chat stream failed')
}

async function streamChat(payload, handlers = {}) {
  const response = await fetch(`${API_BASE}/chat/stream`, {
    method: 'POST',
    headers: { 'Content-Type': 'application/json' },
    body: JSON.stringify(payload),
  })
  if (!response.ok) {
    let detail = `${response.status} ${response.statusText}`
    try {
      const data = await response.json()
      detail = data.detail || detail
    } catch {
      // ignore JSON parse errors for non-JSON error bodies
    }
    throw new Error(detail)
  }
  if (!response.body) throw new Error('Streaming is not supported in this browser')

  const reader = response.body.getReader()
  const decoder = new TextDecoder()
  let buffer = ''

  while (true) {
    const { done, value } = await reader.read()
    if (done) break
    buffer += decoder.decode(value, { stream: true })
    let boundary = buffer.indexOf('\n\n')
    while (boundary !== -1) {
      const block = buffer.slice(0, boundary)
      buffer = buffer.slice(boundary + 2)
      if (block.trim()) dispatchSSEBlock(block, handlers)
      boundary = buffer.indexOf('\n\n')
    }
  }
  if (buffer.trim()) dispatchSSEBlock(buffer, handlers)
}

export default App
