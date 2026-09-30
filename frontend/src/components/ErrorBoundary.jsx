import React from "react";

export default class ErrorBoundary extends React.Component {
  constructor(props) {
    super(props);
    this.state = { hasError: false, err: null };
  }
  static getDerivedStateFromError(err) {
    return { hasError: true, err };
  }
  componentDidCatch(err, info) {
    console.error("UI ErrorBoundary caught:", err, info);
  }
  render() {
    if (!this.state.hasError) return this.props.children;
    return (
      <div className="page">
        <div className="container">
          <div className="card">
            <h2 className="h2">UI Error</h2>
            <p className="muted">Open DevTools Console for details.</p>
            <pre className="code">{String(this.state.err?.message || this.state.err)}</pre>
            <button className="btn" onClick={() => window.location.reload()}>Reload</button>
          </div>
        </div>
      </div>
    );
  }
}
