/* ============================================================
   Loan Approval Prediction System - Main JavaScript
   Handles: form submission loading state, toast init,
            client-side validation helpers, chart refresh.
   ============================================================ */

document.addEventListener("DOMContentLoaded", function () {

    // ---------------------------------------------------------
    // Initialize all Bootstrap toasts on the page
    // ---------------------------------------------------------
    var toastElList = document.querySelectorAll(".toast");
    toastElList.forEach(function (toastEl) {
        var toast = new bootstrap.Toast(toastEl);
        toast.show();
    });

    // ---------------------------------------------------------
    // Show global loading spinner on loan application form submit
    // ---------------------------------------------------------
    var loanForm = document.getElementById("loanApplicationForm");
    var loader = document.getElementById("globalLoader");

    if (loanForm) {
        loanForm.addEventListener("submit", function (e) {
            if (loanForm.checkValidity()) {
                if (loader) {
                    loader.classList.remove("d-none");
                }
            }
        });
    }

    // ---------------------------------------------------------
    // Bootstrap client-side validation styling
    // ---------------------------------------------------------
    var formsToValidate = document.querySelectorAll(".needs-validation");
    Array.prototype.slice.call(formsToValidate).forEach(function (form) {
        form.addEventListener("submit", function (event) {
            if (!form.checkValidity()) {
                event.preventDefault();
                event.stopPropagation();
                if (loader) {
                    loader.classList.add("d-none");
                }
            }
            form.classList.add("was-validated");
        }, false);
    });

    // ---------------------------------------------------------
    // Refresh charts button (admin dashboard)
    // ---------------------------------------------------------
    var refreshBtn = document.getElementById("refreshChartsBtn");
    if (refreshBtn) {
        refreshBtn.addEventListener("click", function () {
            var originalText = refreshBtn.innerHTML;
            refreshBtn.disabled = true;
            refreshBtn.innerHTML = '<span class="spinner-border spinner-border-sm me-2"></span>Refreshing...';

            fetch("/admin/refresh-charts", { method: "POST" })
                .then(function (res) { return res.json(); })
                .then(function (data) {
                    if (data.status === "success") {
                        showToast("Charts refreshed successfully!", "success");
                        setTimeout(function () { window.location.reload(); }, 900);
                    } else {
                        showToast("Failed to refresh charts: " + data.message, "danger");
                        refreshBtn.disabled = false;
                        refreshBtn.innerHTML = originalText;
                    }
                })
                .catch(function (err) {
                    showToast("Error refreshing charts: " + err, "danger");
                    refreshBtn.disabled = false;
                    refreshBtn.innerHTML = originalText;
                });
        });
    }

    // ---------------------------------------------------------
    // Animate probability ring on result page (if present)
    // ---------------------------------------------------------
    var ring = document.querySelector(".probability-ring-fill");
    if (ring) {
        var pct = parseFloat(ring.getAttribute("data-percentage")) || 0;
        var circumference = 2 * Math.PI * 70; // r=70
        var offset = circumference - (pct / 100) * circumference;
        ring.style.strokeDasharray = circumference;
        ring.style.strokeDashoffset = circumference; // start empty
        setTimeout(function () {
            ring.style.transition = "stroke-dashoffset 1.2s ease-out";
            ring.style.strokeDashoffset = offset;
        }, 150);
    }
});


/**
 * Dynamically create and show a Bootstrap toast notification.
 * @param {string} message - The message to display.
 * @param {string} type - Bootstrap contextual color (success, danger, warning, info).
 */
function showToast(message, type) {
    type = type || "info";
    var container = document.querySelector(".toast-container");
    if (!container) {
        container = document.createElement("div");
        container.className = "toast-container position-fixed top-0 end-0 p-3";
        container.style.zIndex = "1080";
        document.body.appendChild(container);
    }

    var toastEl = document.createElement("div");
    toastEl.className = "toast align-items-center text-bg-" + type + " border-0 show";
    toastEl.setAttribute("role", "alert");
    toastEl.innerHTML =
        '<div class="d-flex">' +
            '<div class="toast-body"><i class="bi bi-info-circle-fill me-2"></i>' + message + '</div>' +
            '<button type="button" class="btn-close btn-close-white me-2 m-auto" data-bs-dismiss="toast"></button>' +
        '</div>';

    container.appendChild(toastEl);
    var toast = new bootstrap.Toast(toastEl, { delay: 5000 });
    toast.show();

    toastEl.addEventListener("hidden.bs.toast", function () {
        toastEl.remove();
    });
}
