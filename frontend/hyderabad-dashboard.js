(() => {
    "use strict";

    const SVG_NS = "http://www.w3.org/2000/svg";
    const dashboardStatus = document.getElementById("dashboard-status");
    let dailyObservations = [];

    function fetchJson(url) {
        return fetch(url, { headers: { Accept: "application/json" } }).then((response) => {
            if (!response.ok) {
                throw new Error(`Request failed (${response.status}) for ${url}`);
            }
            return response.json();
        });
    }

    function setText(id, value) {
        document.getElementById(id).textContent = value;
    }

    function formatNumber(value, digits = 2) {
        const number = Number(value);
        if (!Number.isFinite(number)) {
            throw new Error("The analytics response contains an invalid number.");
        }
        return number.toLocaleString("en-IN", {
            minimumFractionDigits: digits,
            maximumFractionDigits: digits
        });
    }

    function formatDate(value) {
        const date = new Date(`${value}T00:00:00Z`);
        if (Number.isNaN(date.getTime())) {
            throw new Error("The analytics response contains an invalid date.");
        }
        return new Intl.DateTimeFormat("en-IN", {
            day: "numeric",
            month: "short",
            year: "numeric",
            timeZone: "UTC"
        }).format(date);
    }

    function setSectionError(messageId, message) {
        const element = document.getElementById(messageId);
        element.textContent = message;
        element.hidden = false;
    }

    function renderSummary(data) {
        setText("summary-year", String(data.dataset_year));
        setText("summary-days", Number(data.observations).toLocaleString("en-IN"));
        setText("summary-mean", formatNumber(data.mean_daily_co2_tonnes));
        setText("summary-min", formatNumber(data.minimum_daily_co2_tonnes));
        setText("summary-max", formatNumber(data.maximum_daily_co2_tonnes));
    }

    function renderLatest(data) {
        setText("latest-date", formatDate(data.date));
        setText("latest-co2", formatNumber(data.co2_tonnes));
    }

    function renderModel(data) {
        const modelName = data.model || data.algorithm;
        setText("model-name", modelName);
        setText("performance-model", modelName);
        setText(
            "forecast-description",
            `${data.forecast_type} based on ${data.dataset_year} observations.`
        );

        if (!data.metrics) {
            throw new Error("Model evaluation metrics are unavailable.");
        }
        setText("metric-mae", formatNumber(data.metrics.mae));
        setText("metric-rmse", formatNumber(data.metrics.rmse));
        setText("metric-r2", formatNumber(data.metrics.r2, 4));
        setText("metric-mape", `${formatNumber(data.metrics.mape_percent)}%`);
    }

    function drawDailyTrend(observations) {
        const canvas = document.getElementById("daily-trend-chart");
        const message = document.getElementById("trend-message");
        const values = observations
            .map((row) => ({ date: row.date, value: Number(row.co2_tonnes) }))
            .filter((row) => Number.isFinite(row.value));

        if (values.length < 2) {
            throw new Error("Not enough daily observations to draw a trend.");
        }

        const bounds = canvas.getBoundingClientRect();
        const ratio = window.devicePixelRatio || 1;
        const width = Math.max(bounds.width, 280);
        const height = Math.max(bounds.height, 200);
        canvas.width = Math.round(width * ratio);
        canvas.height = Math.round(height * ratio);

        const context = canvas.getContext("2d");
        if (!context) {
            throw new Error("The daily trend chart could not be rendered.");
        }
        context.setTransform(ratio, 0, 0, ratio, 0, 0);
        context.clearRect(0, 0, width, height);

        const padding = { top: 14, right: 16, bottom: 34, left: 58 };
        const plotWidth = width - padding.left - padding.right;
        const plotHeight = height - padding.top - padding.bottom;
        const minValue = Math.min(...values.map((row) => row.value));
        const maxValue = Math.max(...values.map((row) => row.value));
        const spread = Math.max(maxValue - minValue, Math.abs(maxValue) * 0.04, 1);
        const lower = minValue - spread * 0.08;
        const upper = maxValue + spread * 0.08;
        const pointX = (index) => padding.left + (index / (values.length - 1)) * plotWidth;
        const pointY = (value) => padding.top + ((upper - value) / (upper - lower)) * plotHeight;

        context.font = "11px DM Sans, sans-serif";
        context.textBaseline = "middle";
        context.strokeStyle = "#e8eef0";
        context.fillStyle = "#71828b";
        context.lineWidth = 1;

        for (let tick = 0; tick <= 4; tick += 1) {
            const y = padding.top + (tick / 4) * plotHeight;
            const value = upper - (tick / 4) * (upper - lower);
            context.beginPath();
            context.moveTo(padding.left, y);
            context.lineTo(width - padding.right, y);
            context.stroke();
            context.textAlign = "right";
            context.fillText(value.toFixed(0), padding.left - 9, y);
        }

        const gradient = context.createLinearGradient(0, padding.top, 0, height - padding.bottom);
        gradient.addColorStop(0, "rgba(8, 126, 120, 0.18)");
        gradient.addColorStop(1, "rgba(8, 126, 120, 0.01)");
        context.beginPath();
        values.forEach((row, index) => {
            const x = pointX(index);
            const y = pointY(row.value);
            if (index === 0) context.moveTo(x, y);
            else context.lineTo(x, y);
        });
        context.lineTo(pointX(values.length - 1), height - padding.bottom);
        context.lineTo(pointX(0), height - padding.bottom);
        context.closePath();
        context.fillStyle = gradient;
        context.fill();

        context.beginPath();
        values.forEach((row, index) => {
            const x = pointX(index);
            const y = pointY(row.value);
            if (index === 0) context.moveTo(x, y);
            else context.lineTo(x, y);
        });
        context.strokeStyle = "#087e78";
        context.lineWidth = 2.2;
        context.lineJoin = "round";
        context.lineCap = "round";
        context.stroke();

        context.fillStyle = "#71828b";
        context.textBaseline = "top";
        context.textAlign = "left";
        context.fillText(formatDate(values[0].date), padding.left, height - padding.bottom + 12);
        context.textAlign = "right";
        context.fillText(formatDate(values[values.length - 1].date), width - padding.right, height - padding.bottom + 12);
        canvas.setAttribute(
            "aria-label",
            `Daily road-traffic CO2 emissions from ${formatDate(values[0].date)} to ${formatDate(values[values.length - 1].date)}`
        );
        message.hidden = true;
    }

    function svgElement(name, attributes = {}, text) {
        const element = document.createElementNS(SVG_NS, name);
        Object.entries(attributes).forEach(([key, value]) => element.setAttribute(key, String(value)));
        if (text !== undefined) element.textContent = text;
        return element;
    }

    function drawHotspotMap(hotspots) {
        const svg = document.getElementById("hotspot-map");
        const message = document.getElementById("map-message");
        const valid = hotspots.filter((point) =>
            Number.isFinite(Number(point.latitude)) &&
            Number.isFinite(Number(point.longitude)) &&
            Number.isFinite(Number(point.mean_co2_tonnes))
        );
        if (!valid.length) {
            throw new Error("No valid hotspot coordinates were returned.");
        }

        const width = 760;
        const height = 390;
        const margin = { top: 22, right: 36, bottom: 58, left: 68 };
        const minLat = Math.min(...valid.map((point) => Number(point.latitude)));
        const maxLat = Math.max(...valid.map((point) => Number(point.latitude)));
        const minLon = Math.min(...valid.map((point) => Number(point.longitude)));
        const maxLon = Math.max(...valid.map((point) => Number(point.longitude)));
        const latPadding = Math.max((maxLat - minLat) * 0.18, 0.004);
        const lonPadding = Math.max((maxLon - minLon) * 0.18, 0.004);
        const south = minLat - latPadding;
        const north = maxLat + latPadding;
        const west = minLon - lonPadding;
        const east = maxLon + lonPadding;
        const plotWidth = width - margin.left - margin.right;
        const plotHeight = height - margin.top - margin.bottom;
        const x = (longitude) => margin.left + ((longitude - west) / (east - west)) * plotWidth;
        const y = (latitude) => margin.top + ((north - latitude) / (north - south)) * plotHeight;
        const means = valid.map((point) => Number(point.mean_co2_tonnes));
        const minMean = Math.min(...means);
        const maxMean = Math.max(...means);

        svg.replaceChildren();
        for (let index = 0; index <= 4; index += 1) {
            const fraction = index / 4;
            const gridX = margin.left + fraction * plotWidth;
            const gridY = margin.top + fraction * plotHeight;
            const longitude = west + fraction * (east - west);
            const latitude = north - fraction * (north - south);
            svg.append(
                svgElement("line", { x1: gridX, y1: margin.top, x2: gridX, y2: height - margin.bottom, class: "map-gridline" }),
                svgElement("line", { x1: margin.left, y1: gridY, x2: width - margin.right, y2: gridY, class: "map-gridline" }),
                svgElement("text", { x: gridX, y: height - margin.bottom + 20, class: "map-axis-label", "text-anchor": "middle" }, longitude.toFixed(4)),
                svgElement("text", { x: margin.left - 10, y: gridY + 4, class: "map-axis-label", "text-anchor": "end" }, latitude.toFixed(4))
            );
        }

        svg.append(
            svgElement("text", { x: margin.left + plotWidth / 2, y: height - 10, class: "map-axis-title", "text-anchor": "middle" }, "Longitude"),
            svgElement("text", {
                x: 17, y: margin.top + plotHeight / 2, class: "map-axis-title",
                "text-anchor": "middle", transform: `rotate(-90 17 ${margin.top + plotHeight / 2})`
            }, "Latitude")
        );

        valid.forEach((point, index) => {
            const mean = Number(point.mean_co2_tonnes);
            const fraction = maxMean === minMean ? 0.5 : (mean - minMean) / (maxMean - minMean);
            const marker = svgElement("circle", {
                cx: x(Number(point.longitude)),
                cy: y(Number(point.latitude)),
                r: 6 + fraction * 6,
                fill: fraction > 0.65 ? "#c65b3a" : fraction > 0.3 ? "#d89a3d" : "#168d83",
                class: "hotspot-marker"
            });
            marker.appendChild(svgElement(
                "title",
                {},
                `Rank ${point.rank || index + 1} · ${Number(point.latitude).toFixed(5)}, ${Number(point.longitude).toFixed(5)} · mean ${formatNumber(mean)} tonnes/day`
            ));
            svg.appendChild(marker);
        });
        svg.setAttribute("aria-label", `Map plot with ${valid.length} validated spatial emission hotspots`);
        message.hidden = true;
    }

    function renderHotspots(data) {
        const rows = document.getElementById("hotspot-rows");
        const hotspots = Array.isArray(data.hotspots) ? data.hotspots.slice(0, 10) : [];
        rows.replaceChildren();
        if (!hotspots.length) {
            const emptyRow = document.createElement("tr");
            const cell = document.createElement("td");
            cell.colSpan = 6;
            cell.className = "table-message";
            cell.textContent = "No validated hotspots were returned.";
            emptyRow.appendChild(cell);
            rows.appendChild(emptyRow);
            setSectionError("map-message", "No validated hotspot coordinates were returned.");
            return;
        }

        drawHotspotMap(hotspots);
        hotspots.forEach((point, index) => {
            const row = document.createElement("tr");
            const values = [
                point.rank ?? index + 1,
                formatNumber(point.latitude, 5),
                formatNumber(point.longitude, 5),
                formatNumber(point.mean_co2_tonnes),
                formatNumber(point.max_daily_co2_tonnes),
                `${formatNumber(point.coverage_percent, 1)}%`
            ];
            values.forEach((value) => {
                const cell = document.createElement("td");
                cell.textContent = String(value);
                row.appendChild(cell);
            });
            rows.appendChild(row);
        });
    }

    async function loadDashboard() {
        const results = await Promise.allSettled([
            fetchJson("/api/hyderabad/summary"),
            fetchJson("/api/hyderabad/latest"),
            fetchJson("/api/hyderabad/model"),
            fetchJson("/api/hyderabad/hotspots?limit=10"),
            fetchJson("/api/hyderabad/daily")
        ]);

        const [summary, latest, model, hotspots, daily] = results;
        if (summary.status === "fulfilled") {
            try {
                renderSummary(summary.value);
                dashboardStatus.hidden = true;
            } catch (error) {
                dashboardStatus.classList.add("is-error");
                dashboardStatus.textContent = "Unable to load Hyderabad analytics data.";
                console.error("Unable to render Hyderabad summary:", error);
            }
        } else {
            dashboardStatus.classList.add("is-error");
            dashboardStatus.textContent = "Unable to load Hyderabad analytics data.";
            console.error("Unable to load Hyderabad summary:", summary.reason);
        }

        if (latest.status === "fulfilled") {
            try {
                renderLatest(latest.value);
            } catch (error) {
                setText("latest-date", "Unavailable");
                setText("latest-co2", "Unavailable");
                console.error("Unable to render latest Hyderabad observation:", error);
            }
        } else {
            setText("latest-date", "Unavailable");
            setText("latest-co2", "Unavailable");
            console.error("Unable to load latest Hyderabad observation:", latest.reason);
        }

        if (model.status === "fulfilled") {
            try {
                renderModel(model.value);
            } catch (error) {
                setText("model-name", "Unavailable");
                setText("performance-model", "Unavailable");
                setText("forecast-description", "Model details are unavailable.");
                ["metric-mae", "metric-rmse", "metric-r2", "metric-mape"].forEach((id) => setText(id, "Unavailable"));
                console.error("Unable to render Hyderabad model details:", error);
            }
        } else {
            setText("model-name", "Unavailable");
            setText("performance-model", "Unavailable");
            setText("forecast-description", "Model details are unavailable.");
            ["metric-mae", "metric-rmse", "metric-r2", "metric-mape"].forEach((id) => setText(id, "Unavailable"));
            console.error("Unable to load Hyderabad model details:", model.reason);
        }

        if (hotspots.status === "fulfilled") {
            try {
                renderHotspots(hotspots.value);
            } catch (error) {
                setSectionError("map-message", "Unable to render spatial emission hotspots.");
                const rows = document.getElementById("hotspot-rows");
                rows.replaceChildren();
                const row = document.createElement("tr");
                const cell = document.createElement("td");
                cell.colSpan = 6;
                cell.className = "table-message";
                cell.textContent = "Unable to render spatial emission hotspots.";
                row.appendChild(cell);
                rows.appendChild(row);
                console.error("Unable to render Hyderabad hotspots:", error);
            }
        } else {
            setSectionError("map-message", "Unable to load spatial emission hotspots.");
            const rows = document.getElementById("hotspot-rows");
            rows.replaceChildren();
            const row = document.createElement("tr");
            const cell = document.createElement("td");
            cell.colSpan = 6;
            cell.className = "table-message";
            cell.textContent = "Unable to load validated hotspots.";
            row.appendChild(cell);
            rows.appendChild(row);
            console.error("Unable to load Hyderabad hotspots:", hotspots.reason);
        }

        if (daily.status === "fulfilled") {
            dailyObservations = Array.isArray(daily.value.observations) ? daily.value.observations : [];
            try {
                drawDailyTrend(dailyObservations);
            } catch (error) {
                setSectionError("trend-message", "Unable to render the daily CO2 trend.");
                console.error("Unable to render Hyderabad daily trend:", error);
            }
        } else {
            setSectionError("trend-message", "Unable to load daily CO2 observations.");
            console.error("Unable to load Hyderabad daily observations:", daily.reason);
        }
    }

    window.addEventListener("resize", () => {
        if (!dailyObservations.length) return;
        window.requestAnimationFrame(() => {
            try {
                drawDailyTrend(dailyObservations);
            } catch (error) {
                setSectionError("trend-message", "Unable to render the daily CO2 trend.");
                console.error("Unable to redraw Hyderabad daily trend:", error);
            }
        });
    });

    loadDashboard();
})();
