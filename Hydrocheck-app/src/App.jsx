import React, { useState, useEffect, useRef } from "react";
import { BleClient } from "@capacitor-community/bluetooth-le";

const SERVICE_UUID = "0000181a-0000-1000-8000-00805f9b34fb";
const CHARACTERISTIC_UUID = "00002a6e-0000-1000-8000-00805f9b34fb";
const RING_CIRCUMFERENCE = 2 * Math.PI * 50;

function BpmChart({ bpmHistory }) {
	const canvasRef = useRef(null);

	useEffect(() => {
		const canvas = canvasRef.current;
		if (!canvas) return;
		const ctx = canvas.getContext("2d");

		const handleResize = () => {
			const w = (canvas.width = canvas.parentElement.clientWidth);
			const h = (canvas.height = canvas.parentElement.clientHeight);

			const margin = 16;
			const chartW = w - margin * 2;
			const chartH = h - margin * 2;

			const drawGrid = () => {
				ctx.strokeStyle = "#272830";
				ctx.lineWidth = 1;
				ctx.setLineDash([5, 5]);
				const steps = 3;
				for (let i = 0; i <= steps; i++) {
					const y = margin + (i / steps) * chartH;
					ctx.beginPath();
					ctx.moveTo(margin, y);
					ctx.lineTo(w - margin, y);
					ctx.stroke();
				}
				ctx.setLineDash([]);
			};

			ctx.clearRect(0, 0, w, h);

			if (bpmHistory.length === 0) {
				drawGrid();
				ctx.fillStyle = "#7d7e84";
				ctx.font = "12px sans-serif";
				ctx.textAlign = "center";
				ctx.textBaseline = "middle";
				ctx.fillText("Awaiting live telemetry...", w / 2, h / 2);
				return;
			}

			drawGrid();

			let minVal = Math.min(...bpmHistory) - 5;
			let maxVal = Math.max(...bpmHistory) + 5;
			if (minVal === maxVal) {
				minVal -= 10;
				maxVal += 10;
			}
			const valRange = maxVal - minVal;

			ctx.beginPath();
			ctx.strokeStyle = "#ff4d4d";
			ctx.lineWidth = 3;
			ctx.lineCap = "round";
			ctx.lineJoin = "round";

			const points = [];
			for (let i = 0; i < bpmHistory.length; i++) {
				const val = bpmHistory[i];
				const x =
					margin + (i / Math.max(1, bpmHistory.length - 1)) * chartW;
				const y = margin + chartH - ((val - minVal) / valRange) * chartH;
				points.push({ x, y });
			}

			if (points.length > 0) {
				ctx.moveTo(points[0].x, points[0].y);
				for (let i = 1; i < points.length; i++) {
					ctx.lineTo(points[i].x, points[i].y);
				}
				ctx.stroke();
			}

			for (let i = 0; i < points.length; i++) {
				const { x, y } = points[i];
				ctx.fillStyle = "#ff4d4d";
				ctx.beginPath();
				ctx.arc(x, y, 5, 0, Math.PI * 2);
				ctx.fill();

				ctx.fillStyle = "#ffffff";
				ctx.beginPath();
				ctx.arc(x, y, 2, 0, Math.PI * 2);
				ctx.fill();

				if (i === points.length - 1) {
					ctx.fillStyle = "#ffffff";
					ctx.font = "bold 10px sans-serif";
					ctx.textAlign = "center";
					ctx.fillText(Math.round(bpmHistory[i]), x, y - 10);
				}
			}
		};

		handleResize();
		window.addEventListener("resize", handleResize);
		return () => window.removeEventListener("resize", handleResize);
	}, [bpmHistory]);

	return <canvas ref={canvasRef} className="w-full h-full block" />;
}

export default function App() {
	const [isConnected, setIsConnected] = useState(false);
	const [statusText, setStatusText] = useState("Not connected");
	const [bluetoothSupported, setBluetoothSupported] = useState(true);
	const [errorTip, setErrorTip] = useState(null);

	const [risk, setRisk] = useState(null);
	const [hr, setHr] = useState(null);
	const [skin, setSkin] = useState(null);
	const [strain, setStrain] = useState(null);
	const [gsr, setGsr] = useState(null);

	const [bpmHistory, setBpmHistory] = useState([]);
	const [logs, setLogs] = useState([]);

	const deviceIdRef = useRef(null);

	const addLog = (msg) => {
		const time = new Date().toLocaleTimeString();
		setLogs((prev) =>
			[{ id: Math.random().toString(), time, msg }, ...prev].slice(0, 50),
		);
	};

	useEffect(() => {
		const initBle = async () => {
			try {
				await BleClient.initialize();
				setBluetoothSupported(true);
				addLog("Bluetooth LE initialized successfully.");
			} catch (e) {
				console.error("BLE init failed", e);
				setBluetoothSupported(false);
				setStatusText("Bluetooth LE not available");
				addLog("Bluetooth LE initialization failed. Platform may not support Bluetooth.");
			}
		};
		initBle();
	}, []);

	const handleNotificationValue = (value) => {
		const decoder = new TextDecoder("utf-8");
		const text = decoder.decode(value.buffer);
		try {
			const data = JSON.parse(text);

			const newRisk = Math.max(0, Math.min(100, data.risk ?? 0));
			setRisk(newRisk);

			if (data.hr !== undefined) {
				const roundedHr = Math.round(data.hr);
				setHr(roundedHr);
				setBpmHistory((prev) => {
					const next = [...prev, roundedHr];
					if (next.length > 8) next.shift();
					return next;
				});
			}

			if (data.skin !== undefined) setSkin(data.skin);
			if (data.strain !== undefined) setStrain(Math.round(data.strain));
			if (data.gsr !== undefined) setGsr(Math.round(data.gsr));

			addLog(
				`risk ${Math.round(newRisk)}% · hr ${Math.round(data.hr ?? 0)}bpm · skin ${(data.skin ?? 0).toFixed(1)}°C`,
			);
		} catch (err) {
			addLog(`parse error: ${text}`);
		}
	};

	const handleDisconnection = () => {
		addLog("Device disconnected.");
		setIsConnected(false);
		setStatusText("Not connected");
		setRisk(null);
		setHr(null);
		setSkin(null);
		setStrain(null);
		setGsr(null);
		deviceIdRef.current = null;
	};

	const connect = async () => {
		console.log("Connect function called");
		setErrorTip(null);
		try {
			addLog("Requesting device...");
			const device = await BleClient.requestDevice({
				services: [SERVICE_UUID],
				name: "Hydrocheck",
			});

			deviceIdRef.current = device.deviceId;
			addLog(`Found ${device.name || "Hydrocheck"}, connecting...`);

			await BleClient.connect(device.deviceId, handleDisconnection);
			addLog("GATT server connected successfully.");

			await BleClient.startNotifications(
				device.deviceId,
				SERVICE_UUID,
				CHARACTERISTIC_UUID,
				handleNotificationValue
			);

			setIsConnected(true);
			setStatusText("Streaming live");
			addLog("Subscribed to live notifications.");
		} catch (err) {
			console.error(err);
			addLog(`Connection failed: ${err.message || err}`);
			setIsConnected(false);
			setStatusText("Not connected");

			if (
				err.message?.includes("permission") ||
				err.name === "SecurityError"
			) {
				setErrorTip(
					"Bluetooth permission is blocked. Please ensure Bluetooth permissions are granted to the application.",
				);
			}
		}
	};

	// Determine styling color based on risk levels
	let riskColor = "var(--cool)";
	let riskWord = "Awaiting data";
	let riskSub =
		"Connect to the wristband to start streaming live physiological readings.";

	if (risk !== null) {
		if (risk > 60) {
			riskColor = "var(--hot)";
			riskWord = "High risk";
			riskSub =
				"Elevated dehydration / heat strain signals detected. Rest and hydrate advised.";
		} else if (risk > 30) {
			riskColor = "var(--warn)";
			riskWord = "Moderate risk";
			riskSub = "Some deviation from resting baseline. Keep monitoring.";
		} else {
			riskColor = "var(--cool)";
			riskWord = "Low risk";
			riskSub = "Physiological readings are within normal range.";
		}
	}

	const dashOffset = risk !== null ? (risk / 100) * RING_CIRCUMFERENCE : 0;

	return (
		<div className="w-full max-w-150 mx-auto px-4 md:px-0 flex flex-col items-center">
			<header className="flex justify-between items-center w-full mb-7">
				<div className="flex items-center gap-3">
					<div className="w-9 h-9">
						<svg
							className="w-full h-full block"
							viewBox="0 0 24 24"
							fill="none"
							xmlns="http://www.w3.org/2000/svg"
						>
							<defs>
								<linearGradient
									id="logoGrad"
									x1="0%"
									y1="0%"
									x2="100%"
									y2="100%"
								>
									<stop offset="0%" stopColor="#3ecf8e" />
									<stop offset="100%" stopColor="#00d2ff" />
								</linearGradient>
							</defs>
							<path
								d="M12 2C12 2 6 8.5 6 13C6 16.3137 8.68629 19 12 19C15.3137 19 18 16.3137 18 13C18 8.5 12 2 12 2Z"
								stroke="url(#logoGrad)"
								strokeWidth="2.5"
								strokeLinejoin="round"
								fill="rgba(62, 207, 142, 0.05)"
							/>
							<path
								d="M9 13.5L11 15.5L15 10.5"
								stroke="#ffffff"
								strokeWidth="2.5"
								strokeLinecap="round"
								strokeLinejoin="round"
							/>
						</svg>
					</div>
					<div>
						<h1 className="text-xl md:text-2xl font-bold tracking-tight">
							Hydrocheck
						</h1>
						<span className="block text-xs text-[#7d7e84] mt-1">
							{statusText}
						</span>
					</div>
				</div>
				<button
					className={`px-4.5 py-2.5 rounded-full font-semibold text-xs transition duration-200 active:scale-95 disabled:opacity-50 
            disabled:cursor-not-allowed cursor-pointer ${
					isConnected
						? "bg-[rgba(62,207,142,0.15)] text-[#3ecf8e]"
						: "bg-[#1b1c21] text-white hover:bg-[#272830]"
				}`}
					onClick={connect}
					disabled={!bluetoothSupported || isConnected}
				>
					{isConnected ? "Connected" : "Connect device"}
				</button>
			</header>

			{errorTip && (
				<div className="w-full bg-red-500/10 border border-red-500/20 text-red-200 text-xs rounded-xl p-4 mb-4">
					<span className="font-semibold block mb-1">
						⚠️ Action Required:
					</span>
					{errorTip}
				</div>
			)}

			<main className="w-full flex flex-col gap-4">
				<div className="bg-[#1b1c21] rounded-3xl p-6 md:p-7 grid grid-cols-1 sm:grid-cols-[1fr_120px] gap-5 items-center shadow-lg">
					<div className="flex flex-col gap-2 items-start">
						<div className="text-lg font-semibold text-white">
							Dehydration Risk
						</div>
						<div className="text-sm md:text-base text-[#7d7e84] font-medium">
							Current level:{" "}
							<span className="font-bold text-white">
								{risk !== null ? risk : "--"}
							</span>
							%
						</div>
						<div
							className="bg-[#272830] px-4 py-2 rounded-2xl text-xs font-semibold inline-block mt-2 transition-colors duration-300"
							style={{ color: riskColor }}
						>
							{riskWord}
						</div>
					</div>

					<div className="relative w-[120px] h-[120px] mx-auto">
						<svg
							className="w-full h-full block transform -rotate-90"
							viewBox="0 0 120 120"
						>
							<circle
								className="fill-none stroke-[#272830] stroke-10"
								cx="60"
								cy="60"
								r="50"
							></circle>
							<circle
								className="fill-none stroke-10 stroke-linecap-round transition-all duration-500"
								cx="60"
								cy="60"
								r="50"
								strokeDasharray={`${dashOffset} ${RING_CIRCUMFERENCE}`}
								style={{ stroke: riskColor }}
							></circle>
						</svg>
						<div className="absolute inset-0 flex items-center justify-center text-xl font-bold">
							{risk !== null ? `${risk}%` : "--%"}
						</div>
					</div>
				</div>

				<div className="grid grid-cols-2 gap-3.5">
					<div className="bg-[#1b1c21] rounded-[20px] p-5 flex flex-col gap-3 shadow-md">
						<div className="flex justify-between items-center">
							<span className="text-xs md:text-sm font-medium text-[#7d7e84]">
								Heart rate
							</span>
							<span className="text-lg">❤️</span>
						</div>
						<div className="flex items-baseline gap-1">
							<span className="text-2xl md:text-3xl font-bold text-white">
								{hr !== null ? hr : "--"}
							</span>
							<span className="text-xs text-[#7d7e84] font-medium">
								bpm
							</span>
						</div>
					</div>

					<div className="bg-[#1b1c21] rounded-[20px] p-5 flex flex-col gap-3 shadow-md">
						<div className="flex justify-between items-center">
							<span className="text-xs md:text-sm font-medium text-[#7d7e84]">
								Skin temp
							</span>
							<span className="text-lg">🌡️</span>
						</div>
						<div className="flex items-baseline gap-1">
							<span className="text-2xl md:text-3xl font-bold text-white">
								{skin !== null ? skin.toFixed(1) : "--"}
							</span>
							<span className="text-xs text-[#7d7e84] font-medium">
								°C
							</span>
						</div>
					</div>

					<div className="bg-[#1b1c21] rounded-[20px] p-5 flex flex-col gap-3 shadow-md">
						<div className="flex justify-between items-center">
							<span className="text-xs md:text-sm font-medium text-[#7d7e84]">
								Heat strain
							</span>
							<span className="text-lg">⚡</span>
						</div>
						<div className="flex items-baseline gap-1">
							<span className="text-2xl md:text-3xl font-bold text-white">
								{strain !== null ? strain : "--"}
							</span>
							<span className="text-xs text-[#7d7e84] font-medium">
								%
							</span>
						</div>
					</div>

					<div className="bg-[#1b1c21] rounded-[20px] p-5 flex flex-col gap-3 shadow-md">
						<div className="flex justify-between items-center">
							<span className="text-xs md:text-sm font-medium text-[#7d7e84]">
								GSR
							</span>
							<span className="text-lg">〰️</span>
						</div>
						<div className="flex items-baseline gap-1">
							<span className="text-2xl md:text-3xl font-bold text-white">
								{gsr !== null ? gsr : "--"}
							</span>
							<span className="text-xs text-[#7d7e84] font-medium">
								raw
							</span>
						</div>
					</div>
				</div>

				<div className="bg-[#1b1c21] rounded-[20px] p-5 flex flex-col gap-3 shadow-md w-full">
					<div className="flex justify-between items-center">
						<span className="text-xs md:text-sm font-medium text-[#7d7e84]">
							Heart Rate History
						</span>
						<span className="text-lg">📈</span>
					</div>
					<div className="h-37.5 mt-4 relative w-full">
						<BpmChart bpmHistory={bpmHistory} />
					</div>
				</div>

				<div className="bg-[#1b1c21] rounded-[20px] p-5 mt-2 w-full">
					<h3 className="text-xs md:text-sm font-semibold text-[#7d7e84] mb-3">
						Session log
					</h3>
					<div className="max-h-35 overflow-y-auto flex flex-col-reverse gap-1.5 text-xs text-[#7d7e84] pr-1 scrollbar-thin">
						{logs.map((log) => (
							<div
								key={log.id}
								className="p-2 bg-[#272830] rounded-xl text-white"
							>
								<span className="text-[#3ecf8e] font-semibold mr-2">
									{log.time}
								</span>{" "}
								{log.msg}
							</div>
						))}
					</div>
				</div>
			</main>

			<footer className="w-full text-center mt-8 py-4 border-t border-[#272830] text-[10px] md:text-xs text-[#7d7e84] tracking-wide">
				Hydrocheck · Web Bluetooth
			</footer>
		</div>
	);
}
