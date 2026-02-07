const API_URL = import.meta.env.VITE_API_URL || 'http://localhost:8000'

function authHeaders() {
  const key = localStorage.getItem('branchforge_api_key')
  return key ? { 'X-BranchForge-Key': key } : {}
}

export async function createJob({ spec, heatmapFile, outlineFile }) {
  const form = new FormData()
  form.append('spec', JSON.stringify(spec))
  form.append('heatmap_file', heatmapFile)
  if (outlineFile) form.append('outline_file', outlineFile)

  const res = await fetch(`${API_URL}/api/v1/jobs`, {
    method: 'POST',
    headers: authHeaders(),
    body: form,
  })
  if (!res.ok) {
    const t = await res.text()
    throw new Error(`Job create failed: ${res.status} ${t}`)
  }
  return await res.json()
}

export async function getJob(jobId) {
  const res = await fetch(`${API_URL}/api/v1/jobs/${jobId}`, { headers: authHeaders() })
  if (!res.ok) {
    const t = await res.text()
    throw new Error(`Job status failed: ${res.status} ${t}`)
  }
  return await res.json()
}

export async function createKey(email, plan='free') {
  const res = await fetch(`${API_URL}/api/v1/auth/create_key`, {
    method: 'POST',
    headers: { 'Content-Type': 'application/json', ...authHeaders() },
    body: JSON.stringify({ email, plan }),
  })
  if (!res.ok) {
    const t = await res.text()
    throw new Error(`Create key failed: ${res.status} ${t}`)
  }
  return await res.json()
}

export function downloadUrl(candidateId, artifactKey) {
  const key = localStorage.getItem('branchforge_api_key')
  const q = key ? `?key=${encodeURIComponent(key)}` : ''
  return `${API_URL}/api/v1/candidates/${candidateId}/download/${artifactKey}${q}`
}
