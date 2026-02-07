import React from 'react'
import { Routes, Route, Link } from 'react-router-dom'
import Landing from './pages/Landing.jsx'
import Designer from './pages/Designer.jsx'

export default function App() {
  return (
    <div className="app">
      <header className="header">
        <div className="brand">
          <Link to="/" className="brandLink">BranchForge</Link>
          <span className="tagline">Heat map → STEP in 60 seconds</span>
        </div>
        <nav className="nav">
          <a href="https://example.com" target="_blank" rel="noreferrer" className="navLink">Docs</a>
          <Link to="/app" className="navCta">Open app</Link>
        </nav>
      </header>

      <main className="main">
        <Routes>
          <Route path="/" element={<Landing />} />
          <Route path="/app" element={<Designer />} />
        </Routes>
      </main>

      <footer className="footer">
        <div>© {new Date().getFullYear()} BranchForge • MVP build</div>
      </footer>
    </div>
  )
}
