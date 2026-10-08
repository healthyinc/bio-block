export default function DemoWarning() {
  return (
    <div className="demo-warning">
      <span className="demo-warning-icon">△</span>
      <span>
        In-memory, unauthenticated mode — local demonstration only.
        Data is not persisted and sessions expire after 30 minutes of inactivity.
      </span>
    </div>
  );
}
