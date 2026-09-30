import { X } from "lucide-react";
import { motion, AnimatePresence } from "framer-motion";

export default function Modal({ open, title, children, onClose }) {
  return (
    <AnimatePresence>
      {open ? (
        <motion.div
          className="modalBack"
          initial={{ opacity: 0 }}
          animate={{ opacity: 1 }}
          exit={{ opacity: 0 }}
          onMouseDown={onClose}
        >
          <motion.div
            className="modal"
            initial={{ scale: 0.98, y: 10, opacity: 0 }}
            animate={{ scale: 1, y: 0, opacity: 1 }}
            exit={{ scale: 0.98, y: 10, opacity: 0 }}
            onMouseDown={(e) => e.stopPropagation()}
          >
            <div className="modalHeader">
              <div>
                <div className="modalTitle">{title}</div>
                <div className="muted small">Click outside to close</div>
              </div>
              <button className="iconBtn" onClick={onClose} title="Close">
                <X size={18} />
              </button>
            </div>
            <div className="modalBody">{children}</div>
          </motion.div>
        </motion.div>
      ) : null}
    </AnimatePresence>
  );
}
