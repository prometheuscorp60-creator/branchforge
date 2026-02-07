import React from 'react'
import { Link } from 'react-router-dom'

export default function Landing() {
  return (
    <div>
      <div className="card">
        <div className="badge">Surface-optimized branching • CAD-first • Cold plates</div>
        <div className="h1">Generate branching cold‑plate channels automatically. Export STEP.</div>
        <div className="p">
          BranchForge converts a heat map into a manufacturable branching supply/return network and gives you
          pressure-drop + temperature-uniformity estimates before you ever open CFD.
        </div>
        <div className="kpiRow">
          <div className="kpi"><b>Input</b>: plate outline + heat map + ports</div>
          <div className="kpi"><b>Output</b>: STEP/STL/DXF + PDF report</div>
          <div className="kpi"><b>Cadence</b>: minutes, not weeks</div>
        </div>
        <div style={{marginTop: 14}}>
          <Link to="/app" className="btn">Upload a heat map → get a STEP</Link>
        </div>
      </div>

      <div style={{height: 14}} />

      <div className="cardGrid">
        <div className="card">
          <h3>What this MVP does</h3>
          <ul className="p">
            <li>Generates multiple branching candidates (different leaf counts / layouts)</li>
            <li>Applies Nature-derived junction rules (sprout vs branch, steering morph)</li>
            <li>Exports a 2.5D CNC-friendly channel void in STEP + STL + DXF</li>
            <li>Ranks candidates and outputs a recommended design</li>
          </ul>
        </div>
        <div className="card">
          <h3>What this MVP intentionally does NOT do</h3>
          <ul className="p">
            <li>Full 3D CFD</li>
            <li>Boiling / multiphase</li>
            <li>Package co-design</li>
            <li>Arbitrary 3D conformal channels (v2)</li>
          </ul>
        </div>
      </div>
    </div>
  )
}
