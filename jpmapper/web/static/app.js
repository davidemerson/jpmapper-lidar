/* JPMapper LOS Analyzer — frontend logic */
(function () {
  "use strict";

  // ── State ──────────────────────────────────────────────────────────────
  let map, markerA, markerB, losLine, boundsRect;
  let obstructionMarkers = [];
  let snapMarkers = [];
  let snapLines = [];
  let coverageLayer = null;
  let profileChart = null;
  let clickCount = 0;

  // ── DOM refs ───────────────────────────────────────────────────────────
  const $latA = document.getElementById("lat-a");
  const $lonA = document.getElementById("lon-a");
  const $latB = document.getElementById("lat-b");
  const $lonB = document.getElementById("lon-b");
  const $mastA = document.getElementById("mast-a");
  const $mastB = document.getElementById("mast-b");
  const $freq = document.getElementById("freq");
  const $btnAnalyze = document.getElementById("btn-analyze");
  const $btnClear = document.getElementById("btn-clear");
  const $loading = document.getElementById("loading");
  const $error = document.getElementById("error");
  const $results = document.getElementById("results");
  const $chartContainer = document.getElementById("chart-container");

  // ── Init map ───────────────────────────────────────────────────────────
  function initMap() {
    map = L.map("map").setView([40.68, -73.95], 13);
    L.tileLayer("https://{s}.tile.openstreetmap.org/{z}/{x}/{y}.png", {
      maxZoom: 19,
      attribution: "&copy; OpenStreetMap contributors",
    }).addTo(map);

    map.on("click", onMapClick);

    // Fetch DSM bounds and fit map view
    fetch("/api/bounds")
      .then((r) => r.json())
      .then((b) => {
        const sw = [b.min_lat, b.min_lon];
        const ne = [b.max_lat, b.max_lon];
        map.fitBounds([sw, ne], { padding: [20, 20] });
        boundsRect = L.rectangle([sw, ne], {
          color: "#4a90d9",
          weight: 1,
          fillOpacity: 0.05,
          dashArray: "5,5",
          interactive: false,
        }).addTo(map);
      })
      .catch(() => {});

    // Fetch and render coverage gaps
    fetchCoverage();

    // Coverage toggle
    document.getElementById("show-coverage").addEventListener("change", function () {
      if (coverageLayer) {
        if (this.checked) {
          map.addLayer(coverageLayer);
        } else {
          map.removeLayer(coverageLayer);
        }
      }
    });
  }

  function fetchCoverage() {
    fetch("/api/coverage")
      .then(function (r) {
        if (!r.ok) throw new Error("Coverage fetch failed: " + r.status);
        return r.json();
      })
      .then(function (data) {
        if (!data.cells || data.cells.length === 0) return;

        // Build GeoJSON FeatureCollection from coverage cells
        var features = data.cells.map(function (c) {
          return {
            type: "Feature",
            properties: { coverage_pct: c.coverage_pct },
            geometry: {
              type: "Polygon",
              coordinates: [[
                [c.min_lon, c.min_lat],
                [c.max_lon, c.min_lat],
                [c.max_lon, c.max_lat],
                [c.min_lon, c.max_lat],
                [c.min_lon, c.min_lat],
              ]],
            },
          };
        });

        var geojson = { type: "FeatureCollection", features: features };

        coverageLayer = L.geoJSON(geojson, {
          style: function (feature) {
            var pct = feature.properties.coverage_pct;
            var opacity;
            if (pct < 10) {
              opacity = 0.45;
            } else if (pct < 50) {
              opacity = 0.3;
            } else {
              opacity = 0.15;
            }
            return {
              color: "#dc3545",
              weight: 0.5,
              opacity: 0.3,
              fillColor: "#dc3545",
              fillOpacity: opacity,
            };
          },
          onEachFeature: function (feature, layer) {
            var pct = feature.properties.coverage_pct;
            layer.bindPopup(
              "<b>Coverage gap</b><br>" +
              "Valid data: " + pct + "%<br>" +
              (pct < 10
                ? "No LiDAR data in this area (missing LAS tile)"
                : "Partial LiDAR coverage")
            );
          },
        });

        if (document.getElementById("show-coverage").checked) {
          coverageLayer.addTo(map);
        }
      })
      .catch(function (err) {
        console.warn("Coverage overlay failed:", err);
      });
  }

  // ── Map click handler ──────────────────────────────────────────────────
  function onMapClick(e) {
    const { lat, lng } = e.latlng;
    if (clickCount % 2 === 0) {
      setPointA(lat, lng);
    } else {
      setPointB(lat, lng);
    }
    clickCount++;
    updateAnalyzeButton();
  }

  function setPointA(lat, lon) {
    $latA.value = lat.toFixed(6);
    $lonA.value = lon.toFixed(6);
    if (markerA) {
      markerA.setLatLng([lat, lon]);
    } else {
      markerA = L.marker([lat, lon], {
        draggable: true,
        icon: blueIcon(),
      })
        .addTo(map)
        .bindTooltip("A", { permanent: true, direction: "top", offset: [0, -36] });
      markerA.on("dragend", function () {
        const p = markerA.getLatLng();
        $latA.value = p.lat.toFixed(6);
        $lonA.value = p.lng.toFixed(6);
      });
    }
  }

  function setPointB(lat, lon) {
    $latB.value = lat.toFixed(6);
    $lonB.value = lon.toFixed(6);
    if (markerB) {
      markerB.setLatLng([lat, lon]);
    } else {
      markerB = L.marker([lat, lon], {
        draggable: true,
        icon: redIcon(),
      })
        .addTo(map)
        .bindTooltip("B", { permanent: true, direction: "top", offset: [0, -36] });
      markerB.on("dragend", function () {
        const p = markerB.getLatLng();
        $latB.value = p.lat.toFixed(6);
        $lonB.value = p.lng.toFixed(6);
      });
    }
  }

  function blueIcon() {
    return L.icon({
      iconUrl: "https://unpkg.com/leaflet@1.9.4/dist/images/marker-icon.png",
      shadowUrl: "https://unpkg.com/leaflet@1.9.4/dist/images/marker-shadow.png",
      iconSize: [25, 41], iconAnchor: [12, 41], popupAnchor: [1, -34], shadowSize: [41, 41],
    });
  }

  function redIcon() {
    return L.icon({
      iconUrl: "https://raw.githubusercontent.com/pointhi/leaflet-color-markers/master/img/marker-icon-red.png",
      shadowUrl: "https://unpkg.com/leaflet@1.9.4/dist/images/marker-shadow.png",
      iconSize: [25, 41], iconAnchor: [12, 41], popupAnchor: [1, -34], shadowSize: [41, 41],
    });
  }

  // ── Sync inputs → markers ──────────────────────────────────────────────
  [$latA, $lonA].forEach((el) =>
    el.addEventListener("change", function () {
      const lat = parseFloat($latA.value);
      const lon = parseFloat($lonA.value);
      if (!isNaN(lat) && !isNaN(lon)) {
        setPointA(lat, lon);
        updateAnalyzeButton();
      }
    })
  );
  [$latB, $lonB].forEach((el) =>
    el.addEventListener("change", function () {
      const lat = parseFloat($latB.value);
      const lon = parseFloat($lonB.value);
      if (!isNaN(lat) && !isNaN(lon)) {
        setPointB(lat, lon);
        updateAnalyzeButton();
      }
    })
  );

  // ── Buttons ────────────────────────────────────────────────────────────
  function updateAnalyzeButton() {
    const hasA = $latA.value && $lonA.value;
    const hasB = $latB.value && $lonB.value;
    $btnAnalyze.disabled = !(hasA && hasB);
  }

  $btnAnalyze.addEventListener("click", runAnalysis);
  $btnClear.addEventListener("click", clearAll);

  function clearAll() {
    if (markerA) { map.removeLayer(markerA); markerA = null; }
    if (markerB) { map.removeLayer(markerB); markerB = null; }
    clearMapOverlays();
    $latA.value = ""; $lonA.value = "";
    $latB.value = ""; $lonB.value = "";
    $results.classList.add("hidden");
    $chartContainer.classList.add("hidden");
    $error.classList.add("hidden");
    document.getElementById("snap-notice").classList.add("hidden");
    $btnAnalyze.disabled = true;
    clickCount = 0;
  }

  function clearMapOverlays() {
    if (losLine) { map.removeLayer(losLine); losLine = null; }
    obstructionMarkers.forEach((m) => map.removeLayer(m));
    obstructionMarkers = [];
    snapMarkers.forEach((m) => map.removeLayer(m));
    snapMarkers = [];
    snapLines.forEach((l) => map.removeLayer(l));
    snapLines = [];
  }

  // ── Analysis ───────────────────────────────────────────────────────────
  async function runAnalysis() {
    const body = {
      point_a: { lat: parseFloat($latA.value), lon: parseFloat($lonA.value) },
      point_b: { lat: parseFloat($latB.value), lon: parseFloat($lonB.value) },
      mast_a_height_m: parseFloat($mastA.value) || 0,
      mast_b_height_m: parseFloat($mastB.value) || 0,
      freq_ghz: parseFloat($freq.value),
    };

    $loading.classList.remove("hidden");
    $error.classList.add("hidden");
    $results.classList.add("hidden");
    $btnAnalyze.disabled = true;

    try {
      const res = await fetch("/api/analyze", {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify(body),
      });
      if (!res.ok) {
        const err = await res.json().catch(() => ({ detail: res.statusText }));
        throw new Error(err.detail || "Analysis failed");
      }
      const data = await res.json();
      updateResults(data);
      updateSnapNotice(data);
      updateMapLink(data, body);
      renderProfile(data);
    } catch (err) {
      $error.textContent = err.message;
      $error.classList.remove("hidden");
    } finally {
      $loading.classList.add("hidden");
      $btnAnalyze.disabled = false;
    }
  }

  // ── Results panel ──────────────────────────────────────────────────────
  function updateResults(data) {
    $results.classList.remove("hidden");
    document.getElementById("status-badge").innerHTML = data.clear
      ? '<span class="badge badge-clear">CLEAR</span>'
      : '<span class="badge badge-blocked">BLOCKED</span>';
    document.getElementById("r-distance").textContent = formatDist(data.distance_m);
    document.getElementById("r-clearance").textContent = data.clearance_min_m.toFixed(1) + " m";
    document.getElementById("r-surface-a").textContent = data.surface_height_a_m.toFixed(1) + " m";
    document.getElementById("r-surface-b").textContent = data.surface_height_b_m.toFixed(1) + " m";
    document.getElementById("r-obstructions").textContent = data.obstructions.length;
  }

  function formatDist(m) {
    return m >= 1000 ? (m / 1000).toFixed(2) + " km" : m.toFixed(0) + " m";
  }

  // ── Map overlays ───────────────────────────────────────────────────────
  function updateMapLink(data, body) {
    clearMapOverlays();
    const color = data.clear ? "#28a745" : "#dc3545";
    // Use snapped coordinates for LOS line when a snap occurred
    const losA = data.snap_a
      ? [data.snap_a.snapped_lat, data.snap_a.snapped_lon]
      : [body.point_a.lat, body.point_a.lon];
    const losB = data.snap_b
      ? [data.snap_b.snapped_lat, data.snap_b.snapped_lon]
      : [body.point_b.lat, body.point_b.lon];
    losLine = L.polyline([losA, losB], {
      color: color, weight: 3, dashArray: data.clear ? null : "8,6",
    }).addTo(map);

    data.obstructions.forEach((o) => {
      const m = L.circleMarker([o.lat, o.lon], {
        radius: 6,
        color: "#dc3545",
        fillColor: "#dc3545",
        fillOpacity: 0.8,
      })
        .addTo(map)
        .bindPopup(
          `<b>Obstruction</b><br>Height: ${o.obstruction_height_m} m<br>Severity: ${o.severity}`
        );
      obstructionMarkers.push(m);
    });

    // Snap indicators
    [data.snap_a, data.snap_b].forEach((snap, idx) => {
      if (!snap) return;
      const label = idx === 0 ? "A" : "B";
      // Dashed line from original to snapped position
      const line = L.polyline(
        [[snap.original_lat, snap.original_lon], [snap.snapped_lat, snap.snapped_lon]],
        { color: "#e67e22", weight: 2, dashArray: "4,4", opacity: 0.8 }
      ).addTo(map);
      snapLines.push(line);
      // Diamond marker at snapped position
      const marker = L.circleMarker([snap.snapped_lat, snap.snapped_lon], {
        radius: 7,
        color: "#e67e22",
        fillColor: "#f39c12",
        fillOpacity: 0.9,
        weight: 2,
      })
        .addTo(map)
        .bindPopup(
          `<b>Point ${label} snapped</b><br>` +
          `Moved ${snap.snap_distance_m} m to nearest valid DSM data`
        );
      snapMarkers.push(marker);
    });
  }

  function updateSnapNotice(data) {
    const $snap = document.getElementById("snap-notice");
    const parts = [];
    if (data.snap_a) {
      parts.push(`<strong>Point A</strong> snapped ${data.snap_a.snap_distance_m} m`);
    }
    if (data.snap_b) {
      parts.push(`<strong>Point B</strong> snapped ${data.snap_b.snap_distance_m} m`);
    }
    if (parts.length > 0) {
      $snap.innerHTML = "Snapped to nearest DSM data: " + parts.join(", ");
      $snap.classList.remove("hidden");
    } else {
      $snap.classList.add("hidden");
    }
  }

  // ── Profile chart ──────────────────────────────────────────────────────
  function renderProfile(data) {
    $chartContainer.classList.remove("hidden");
    const p = data.profile;
    const labels = p.distances_m.map((d) => d.toFixed(0));

    // Fresnel upper/lower bounds
    const fresnelUpper = p.los_heights_m.map((h, i) => h + 0.6 * p.fresnel_radii_m[i]);
    const fresnelLower = p.los_heights_m.map((h, i) => h - 0.6 * p.fresnel_radii_m[i]);

    // Obstruction points
    const obstructionData = new Array(p.distances_m.length).fill(null);
    data.obstructions.forEach((o) => {
      // Find closest distance index
      let bestIdx = 0;
      let bestDiff = Infinity;
      for (let i = 0; i < p.distances_m.length; i++) {
        const diff = Math.abs(p.distances_m[i] - o.distance_along_path_m);
        if (diff < bestDiff) { bestDiff = diff; bestIdx = i; }
      }
      obstructionData[bestIdx] = p.terrain_heights_m[bestIdx];
    });

    if (profileChart) profileChart.destroy();

    const ctx = document.getElementById("profile-chart").getContext("2d");
    profileChart = new Chart(ctx, {
      type: "line",
      data: {
        labels: labels,
        datasets: [
          {
            label: "Terrain",
            data: p.terrain_heights_m,
            borderColor: "#8B6914",
            backgroundColor: "rgba(139,105,20,0.25)",
            fill: true,
            pointRadius: 0,
            borderWidth: 1.5,
            order: 3,
          },
          {
            label: "LOS Line",
            data: p.los_heights_m,
            borderColor: "#333",
            borderDash: [6, 4],
            pointRadius: 0,
            borderWidth: 1.5,
            fill: false,
            order: 2,
          },
          {
            label: "60% Fresnel Upper",
            data: fresnelUpper,
            borderColor: "rgba(255,165,0,0.4)",
            backgroundColor: "rgba(255,165,0,0.1)",
            fill: "+1",
            pointRadius: 0,
            borderWidth: 1,
            order: 1,
          },
          {
            label: "60% Fresnel Lower",
            data: fresnelLower,
            borderColor: "rgba(255,165,0,0.4)",
            pointRadius: 0,
            borderWidth: 1,
            fill: false,
            order: 1,
          },
          {
            label: "Obstructions",
            data: obstructionData,
            borderColor: "#dc3545",
            backgroundColor: "#dc3545",
            pointRadius: 6,
            pointStyle: "circle",
            showLine: false,
            order: 0,
          },
        ],
      },
      options: {
        responsive: true,
        maintainAspectRatio: false,
        interaction: { mode: "index", intersect: false },
        plugins: {
          legend: { position: "top", labels: { usePointStyle: true, boxWidth: 8, font: { size: 11 } } },
          tooltip: {
            callbacks: {
              title: function (items) { return "Distance: " + items[0].label + " m"; },
              label: function (item) {
                if (item.raw === null) return null;
                return item.dataset.label + ": " + item.raw.toFixed(1) + " m";
              },
            },
          },
        },
        scales: {
          x: {
            title: { display: true, text: "Distance (m)" },
            ticks: {
              maxTicksLimit: 12,
              callback: function (val, i) {
                return i % Math.ceil(labels.length / 12) === 0 ? labels[i] : "";
              },
            },
          },
          y: { title: { display: true, text: "Elevation (m)" } },
        },
      },
    });
  }

  // ── Boot ───────────────────────────────────────────────────────────────
  initMap();
})();
