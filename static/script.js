/**
 * Drowsiness Detection — Client-side logic
 * Polls /status every 500ms and manages alarm audio + visual alerts.
 */
(function () {
  "use strict";

  const alarmAudio   = document.getElementById("alarm-audio");
  const alertOverlay = document.getElementById("alert-overlay");
  const videoCard    = document.getElementById("video-card");
  const statusDot    = document.getElementById("status-dot");
  const cardTitle    = document.getElementById("card-title");
  const statusText   = document.getElementById("status-text");
  const soundStatus  = document.getElementById("sound-status");
  const muteBtn      = document.getElementById("toggle-sound-btn");

  let muted = false;

  // ── Mute / Unmute toggle ──
  muteBtn.addEventListener("click", () => {
    muted = !muted;
    muteBtn.textContent = muted ? "🔕 Unmute Alarm" : "🔔 Mute Alarm";
    muteBtn.classList.toggle("muted", muted);
    soundStatus.textContent = muted ? "Muted" : "Ready";

    if (muted && !alarmAudio.paused) {
      alarmAudio.pause();
      alarmAudio.currentTime = 0;
    }
  });

  // ── Poll drowsiness status ──
  async function pollStatus() {
    try {
      const res  = await fetch("/status");
      const data = await res.json();

      if (data.drowsy) {
        activateAlert();
      } else {
        deactivateAlert();
      }
    } catch (err) {
      console.error("Status poll error:", err);
    }
  }

  function activateAlert() {
    alertOverlay.classList.add("active");
    videoCard.classList.add("alert-glow");
    statusDot.classList.add("alert");
    cardTitle.textContent = "⚠ DROWSINESS ALERT";
    statusText.textContent = "Drowsy!";
    statusText.className = "stat-value danger";

    if (!muted && alarmAudio.paused) {
      alarmAudio.play().catch(() => {});
    }
  }

  function deactivateAlert() {
    alertOverlay.classList.remove("active");
    videoCard.classList.remove("alert-glow");
    statusDot.classList.remove("alert");
    cardTitle.textContent = "Live Feed";
    statusText.textContent = "Awake ✓";
    statusText.className = "stat-value safe";

    if (!alarmAudio.paused) {
      alarmAudio.pause();
      alarmAudio.currentTime = 0;
    }
  }

  // Start polling
  setInterval(pollStatus, 500);
})();
