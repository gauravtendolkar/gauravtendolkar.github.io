// Live App Store ratings for /apps/.
// The page is rendered with a fallback rating from _data/apps.yml; on load this asks
// Apple's iTunes Lookup API for the current averageUserRating / userRatingCount and
// updates each [data-app-store-id] element. Tries fetch (CORS) first, then JSONP.
(function () {
  "use strict";

  var nodes = document.querySelectorAll(".app-rating[data-app-store-id]");
  if (!nodes.length) return;

  var ids = [];
  nodes.forEach(function (n) {
    var id = n.getAttribute("data-app-store-id");
    if (id && ids.indexOf(id) < 0) ids.push(id);
  });
  var url = "https://itunes.apple.com/lookup?country=us&id=" + ids.join(",");

  function render(data) {
    var byId = {};
    (data && data.results || []).forEach(function (r) { byId[String(r.trackId)] = r; });
    nodes.forEach(function (n) {
      var r = byId[n.getAttribute("data-app-store-id")];
      if (!r) return;
      var avg = Number(r.averageUserRating) || 0;
      var count = Number(r.userRatingCount) || 0;
      if (!count) { n.hidden = true; return; }
      var shown = avg.toFixed(1);
      var stars = n.querySelector(".app-rating__stars");
      stars.style.setProperty("--rating", (avg / 5 * 100) + "%");
      stars.setAttribute("aria-label", "Rated " + shown + " out of 5 on the App Store");
      n.querySelector(".app-rating__value").textContent = shown;
      n.setAttribute("data-rating-source", "live");
      n.hidden = false;
    });
  }

  function jsonp() {
    var cb = "__appRatings" + Date.now();
    var s = document.createElement("script");
    var done = function () { delete window[cb]; s.remove(); };
    window[cb] = function (data) { done(); render(data); };
    s.onerror = done;
    s.src = url + "&callback=" + cb;
    document.head.appendChild(s);
  }

  if (window.fetch) {
    fetch(url).then(function (res) {
      if (!res.ok) throw new Error(res.status);
      return res.json();
    }).then(render).catch(jsonp);
  } else {
    jsonp();
  }
})();
