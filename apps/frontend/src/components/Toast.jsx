const MyToast = ({ toasts, removeToast }) => {

  return <>
    <div
      className={`
        toast-container position-fixed top-0 end-0 p-3
      `}
      style={{ zIndex: 9999 }}
    >
      {toasts.map(toast => (
      <div
        key={toast.id}
        className={`
          toast fade
          ${toast.showing ? 'slide-in' : 'slide-out'}
          text-bg-${toast.color}
          d-flex
          align-items-center
          border-0
        `}
        role="alert"
        aria-live="assertive"
        aria-atomic='true'
      >
        <div className="toast-body">
          {toast.message}
        </div>
        <button
          className="btn-close btn-close-white m-auto me-2"
          type="button"
          onClick={() => removeToast(toast.id)}
        ></button>
      </div>
      ))}
    </div>
  </>;
};

export default MyToast;