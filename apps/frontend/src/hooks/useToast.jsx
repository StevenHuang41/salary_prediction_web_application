import { useState } from "react";

const useToast = () => {
  const [toasts, setToasts] = useState([]);

  const addToast = (message, color) => {
    const id = Date.now() + Math.random();
      // set toast to show model
      setToasts(prev => [
        ...prev,
        {id, message, showing: true, color}
      ]);

      setTimeout(() => {
        // set toast to hide mode, after 3000ms
        setToasts(prev => 
          prev.map(toast => 
            toast.id === id ? { ...toast, showing: false } : toast
          )
        );

        // delete toast from toasts, after 500ms
        setTimeout(() => {
          setToasts(prev => prev.filter(toast => toast.id !== id));
        }, 500);
      }, 3000);
  };

  const removeToast = (id) => {
    setToasts(prev => 
      prev.map(t => 
        t.id === id ? { ...t, showing: false } : t
      )
    );

    setTimeout(() => {
      setToasts(prev => prev.filter(t => t.id !== id));
    }, 500);
  };

  return { toasts, addToast, removeToast };
};

export default useToast;