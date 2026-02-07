import React, { useMemo, useState, useEffect } from 'react'
import { createJob, getJob, downloadUrl, createKey } from '../api.js'

function number(v, fallback) {
  const x = parseFloat(v)
  return Number.isFinite(x) ? x : fallback
}

export default function Designer() {
  const [plateKind, setPlateKind] = useState('rectangle')
  const [plateW, setPlateW] = useState(100)
  const [plateH, setPlateH] = useState(60)
  const [outlineFile, setOutlineFile] = useState(null)

  const [heatmapKind, setHeatmapKind] = useState('csv')
  const [heatmapFile, setHeatmapFile] = useState(null)
  const [totalWatts, setTotalWatts] = useState(1000)

  const [inletX, setInletX] = useState(5)
  const [inletY, setInletY] = useState(30)
  const [outletX, setOutletX] = useState(95)
  const [outletY, setOutletY] = useState(30)

  const [plateTh, setPlateTh] = useState(5)
  const [chDepth, setChDepth] = useState(2)
  const [minChW, setMinChW] = useState(1.2)
  const [maxChW, setMaxChW] = useState(8)
  const [minWall, setMinWall] = useState(1.0)
  const [bendR, setBendR] = useState(3.0)
  const [gridRes, setGridRes] = useState(2.0)

  const [targetDT, setTargetDT] = useState(10)

  const [nCandidates, setNCandidates] = useState(10)
  const [leafCounts, setLeafCounts] = useState('8,12,16')

  const [jobId, setJobId] = useState(null)
  const [job, setJob] = useState(null)
  const [err, setErr] = useState(null)

  const [apiKey, setApiKey] = useState(localStorage.getItem('branchforge_api_key') || '')
  const [email, setEmail] = useState('')
  const [plan, setPlan] = useState('free')

  const spec = useMemo(() => ({
    plate: plateKind === 'rectangle'
      ? { kind: 'rectangle', width_mm: number(plateW, 100), height_mm: number(plateH, 60) }
      : { kind: 'polygon' }, // API will set dxf_path when outline_file provided
    ports: {
      inlet: { x_mm: number(inletX, 5), y_mm: number(inletY, 30) },
      outlet: { x_mm: number(outletX, 95), y_mm: number(outletY, 30) },
      inlet_diameter_mm: 6,
      outlet_diameter_mm: 6,
    },
    heatmap: {
      kind: heatmapKind,
      total_watts: number(totalWatts, 1000),
      flip_y: true,
      // path is set by API after upload
    },
    constraints: {
      plate_thickness_mm: number(plateTh, 5),
      channel_depth_mm: number(chDepth, 2),
      min_channel_width_mm: number(minChW, 1.2),
      max_channel_width_mm: number(maxChW, 8),
      min_wall_mm: number(minWall, 1.0),
      min_bend_radius_mm: number(bendR, 3.0),
      grid_resolution_mm: number(gridRes, 2.0),
      process_preset: 'CNC',
      keepout_rects: [],
      keepout_circles: [],
    },
    fluid: {
      coolant: 'water',
      inlet_temp_C: 25,
      target_deltaT_C: number(targetDT, 10),
    },
    generation: {
      n_candidates: number(nCandidates, 10),
      leaf_counts: leafCounts.split(',').map(s => parseInt(s.trim(), 10)).filter(n => Number.isFinite(n) && n > 1),
      seed: 42,
      weight_pressure: 1.0,
      weight_uniformity: 1.0,
      weight_manufacturing: 0.5,
      v_max_m_per_s: 1.5,
    },
  }), [plateKind, plateW, plateH, inletX, inletY, outletX, outletY, heatmapKind, totalWatts, plateTh, chDepth, minChW, maxChW, minWall, bendR, gridRes, targetDT, nCandidates, leafCounts])


  function onSaveKey() {
    localStorage.setItem('branchforge_api_key', apiKey || '')
    setErr(null)
  }

  async function onCreateKey() {
    setErr(null)
    try {
      const res = await createKey(email, plan)
      setApiKey(res.api_key)
      localStorage.setItem('branchforge_api_key', res.api_key)
    } catch (e) {
      setErr(String(e.message || e))
    }
  }

  async function onGenerate() {
    setErr(null)
    setJob(null)
    setJobId(null)

    if (!heatmapFile) {
      setErr('Please upload a heatmap file.')
      return
    }
    if (plateKind === 'polygon' && !outlineFile) {
      setErr('Please upload a DXF outline.')
      return
    }
    try {
      const res = await createJob({ spec, heatmapFile, outlineFile: plateKind === 'polygon' ? outlineFile : null })
      setJobId(res.job_id)
    } catch (e) {
      setErr(String(e.message || e))
    }
  }

  // Poll job status
  useEffect(() => {
    let timer = null
    let cancelled = false
    async function tick() {
      if (!jobId) return
      try {
        const j = await getJob(jobId)
        if (!cancelled) setJob(j)
        if (j.status === 'succeeded' || j.status === 'failed') return
      } catch (e) {
        if (!cancelled) setErr(String(e.message || e))
      }
      timer = setTimeout(tick, 1500)
    }
    tick()
    return () => { cancelled = true; if (timer) clearTimeout(timer) }
  }, [jobId])

  return (
    <div className="cardGrid">
      <div className="card">
        <h2>Design inputs</h2>

        <div className="card" style={{marginBottom: 12, background: '#fcfcfc'}}>
          <div className="row" style={{justifyContent: 'space-between'}}>
            <div><b>API key</b> <span className="small">(only needed if auth is enabled)</span></div>
          </div>
          <div className="formRow" style={{marginTop: 8}}>
            <div>
              <label>API key</label>
              <input value={apiKey} onChange={e => setApiKey(e.target.value)} placeholder="bf_… or dev" />
              <div className="row" style={{marginTop: 8}}>
                <button className="btn" type="button" onClick={onSaveKey}>Save key</button>
              </div>
            </div>
            <div>
              <label>Create/rotate key (dev)</label>
              <input value={email} onChange={e => setEmail(e.target.value)} placeholder="you@company.com" />
              <div className="row" style={{marginTop: 8}}>
                <select value={plan} onChange={e => setPlan(e.target.value)}>
                  <option value="free">free</option>
                  <option value="pro">pro</option>
                  <option value="team">team</option>
                  <option value="enterprise">enterprise</option>
                </select>
                <button className="btn" type="button" onClick={onCreateKey}>Create key</button>
              </div>
            </div>
          </div>
        </div>

        <div className="formRow">
          <div>
            <label>Plate input</label>
            <select value={plateKind} onChange={e => setPlateKind(e.target.value)}>
              <option value="rectangle">Rectangle (fast)</option>
              <option value="polygon">DXF outline</option>
            </select>
          </div>
          <div>
            <label>Heatmap type</label>
            <select value={heatmapKind} onChange={e => setHeatmapKind(e.target.value)}>
              <option value="csv">CSV grid (watts)</option>
              <option value="image">Image (grayscale intensity)</option>
            </select>
          </div>
        </div>

        {plateKind === 'rectangle' ? (
          <div className="formRow">
            <div>
              <label>Plate width (mm)</label>
              <input value={plateW} onChange={e => setPlateW(e.target.value)} />
            </div>
            <div>
              <label>Plate height (mm)</label>
              <input value={plateH} onChange={e => setPlateH(e.target.value)} />
            </div>
          </div>
        ) : (
          <div>
            <label>DXF outline file</label>
            <input type="file" accept=".dxf" onChange={e => setOutlineFile(e.target.files?.[0] || null)} />
            <div className="small">First closed polyline is used as plate boundary.</div>
          </div>
        )}

        <hr />

        <div className="formRow">
          <div>
            <label>Heatmap file</label>
            <input type="file" accept={heatmapKind === 'csv' ? '.csv,text/csv' : 'image/*'} onChange={e => setHeatmapFile(e.target.files?.[0] || null)} />
            <div className="small">
              CSV: numeric grid of watts per cell (no headers). Image: grayscale intensity scaled to total watts.
            </div>
          </div>
          <div>
            <label>Total watts (W)</label>
            <input value={totalWatts} onChange={e => setTotalWatts(e.target.value)} />
          </div>
        </div>

        <hr />

        <h3>Ports</h3>
        <div className="formRow">
          <div>
            <label>Inlet (x, y) mm</label>
            <div className="formRow">
              <input value={inletX} onChange={e => setInletX(e.target.value)} />
              <input value={inletY} onChange={e => setInletY(e.target.value)} />
            </div>
          </div>
          <div>
            <label>Outlet (x, y) mm</label>
            <div className="formRow">
              <input value={outletX} onChange={e => setOutletX(e.target.value)} />
              <input value={outletY} onChange={e => setOutletY(e.target.value)} />
            </div>
          </div>
        </div>

        <hr />

        <h3>Constraints</h3>
        <div className="formRow">
          <div>
            <label>Plate thickness (mm)</label>
            <input value={plateTh} onChange={e => setPlateTh(e.target.value)} />
          </div>
          <div>
            <label>Channel depth (mm)</label>
            <input value={chDepth} onChange={e => setChDepth(e.target.value)} />
          </div>
        </div>
        <div className="formRow">
          <div>
            <label>Min channel width (mm)</label>
            <input value={minChW} onChange={e => setMinChW(e.target.value)} />
          </div>
          <div>
            <label>Max channel width (mm)</label>
            <input value={maxChW} onChange={e => setMaxChW(e.target.value)} />
          </div>
        </div>
        <div className="formRow">
          <div>
            <label>Min wall (mm)</label>
            <input value={minWall} onChange={e => setMinWall(e.target.value)} />
          </div>
          <div>
            <label>Min bend radius (mm)</label>
            <input value={bendR} onChange={e => setBendR(e.target.value)} />
          </div>
        </div>
        <div className="formRow">
          <div>
            <label>Routing grid resolution (mm)</label>
            <input value={gridRes} onChange={e => setGridRes(e.target.value)} />
          </div>
          <div>
            <label>Target coolant ΔT (°C)</label>
            <input value={targetDT} onChange={e => setTargetDT(e.target.value)} />
          </div>
        </div>

        <hr />

        <h3>Generation</h3>
        <div className="formRow">
          <div>
            <label># candidates</label>
            <input value={nCandidates} onChange={e => setNCandidates(e.target.value)} />
          </div>
          <div>
            <label>Leaf counts (comma separated)</label>
            <input value={leafCounts} onChange={e => setLeafCounts(e.target.value)} />
          </div>
        </div>

        <div style={{marginTop: 14}} className="row">
          <button className="btn" onClick={onGenerate}>Generate</button>
          {jobId && <span className="statusPill">Job: {jobId.slice(0, 8)}…</span>}
          {job && <span className="statusPill">Status: {job.status}</span>}
        </div>
        {err && <div style={{marginTop: 12, color: '#a00', whiteSpace: 'pre-wrap'}}>{err}</div>}
      </div>

      <div className="card">
        <h2>Results</h2>
        {!job && <div className="p">No job yet. Upload inputs and click Generate.</div>}

        {job && job.status !== 'succeeded' && (
          <div className="p">
            {job.status === 'failed' ? (
              <div style={{whiteSpace: 'pre-wrap', color: '#a00'}}>{job.error || 'Failed'}</div>
            ) : (
              <div>Generating candidates… (polling)</div>
            )}
          </div>
        )}

        {job && job.status === 'succeeded' && (
          <div>
            <div className="small">{job.candidates.length} candidates</div>
            <hr />

            {job.candidates.map(c => (
              <div key={c.id} style={{marginBottom: 18}}>
                <div className="row" style={{justifyContent: 'space-between'}}>
                  <div>
                    <b>{c.label}</b> <span className="badge">#{c.index}</span>
                  </div>
                  <div className="small">
                    ΔP {c.metrics.delta_p_kpa?.toFixed?.(2)} kPa • σΔT {c.metrics.uniformity_deltaT_C_std?.toFixed?.(3)} °C • MFG {c.metrics.manufacturing_score?.toFixed?.(2)}
                  </div>
                </div>

                {c.artifacts?.preview_png && (
                  <div style={{marginTop: 10}}>
                    <img className="preview" src={downloadUrl(c.id, 'preview_png')} alt="preview" />
                  </div>
                )}

                <div className="downloadRow" style={{marginTop: 10}}>
                  <a href={downloadUrl(c.id, 'plate_step')} target="_blank" rel="noreferrer">plate.step</a>
                  <a href={downloadUrl(c.id, 'channels_step')} target="_blank" rel="noreferrer">channels.step</a>
                  <a href={downloadUrl(c.id, 'plate_stl')} target="_blank" rel="noreferrer">plate.stl</a>
                  <a href={downloadUrl(c.id, 'channels_dxf')} target="_blank" rel="noreferrer">channels.dxf</a>
                  <a href={downloadUrl(c.id, 'report_pdf')} target="_blank" rel="noreferrer">report.pdf</a>
                  <a href={downloadUrl(c.id, 'json_paths')} target="_blank" rel="noreferrer">paths.json</a>
                </div>

                <hr />
              </div>
            ))}
          </div>
        )}
      </div>
    </div>
  )
}
