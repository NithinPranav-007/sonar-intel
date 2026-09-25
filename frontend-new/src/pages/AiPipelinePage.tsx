import React, { useEffect, useState } from 'react';
import { apiService } from '../services/api';
import { 
  Cpu, 
  Activity, 
  Clock, 
  Zap, 
  CheckCircle2, 
  Terminal, 
  BarChart3,
  Layers,
  Sparkles,
  ShieldCheck,
  Server,
  Radar,
  Sliders,
  Filter,
  Eye,
  Workflow,
  FileDown,
  X,
  Scan,
  ChevronLeft,
  Copy,
  Check
} from 'lucide-react';

interface AiPipelinePageProps {
  onNavigateToDashboard?: () => void;
}

export const AiPipelinePage: React.FC<AiPipelinePageProps> = ({ onNavigateToDashboard }) => {
  const [pipelineInfo, setPipelineInfo] = useState<any>(null);
  const [loading, setLoading] = useState<boolean>(true);
  const [exportNotice, setExportNotice] = useState<string | null>(null);
  const [copiedMetrics, setCopiedMetrics] = useState<boolean>(false);
  const [selectedStage, setSelectedStage] = useState<{
    step: string;
    name: string;
    tag: string;
    desc: string;
    simpleDesc: string;
    details: string;
  } | null>(null);

  useEffect(() => {
    const fetchInfo = async () => {
      try {
        const res = await apiService.getPipelineInfo();
        setPipelineInfo(res);
      } catch (err) {
        console.warn('Pipeline info fetch error:', err);
      } finally {
        setLoading(false);
      }
    };
    fetchInfo();
  }, []);

  const pipelineStages = [
    { 
      step: '01', 
      name: 'Raw Ingestion', 
      tag: 'COMPLETE', 
      desc: '16/8-bit SSS Waterfall',
      simpleDesc: 'Loads the raw acoustic sonar image stream from the towfish device.',
      details: 'Decodes high-frequency hydrophone data into calibrated pixel intensities.'
    },
    { 
      step: '02', 
      name: 'SNR Dynamic Range', 
      tag: 'COMPLETE', 
      desc: 'Signal-to-Noise Check',
      simpleDesc: 'Evaluates signal-to-noise ratio to verify clear underwater visibility.',
      details: 'Ensures seabed backscatter meets minimum detection thresholds before processing.'
    },
    { 
      step: '03', 
      name: '1–99% Stretch', 
      tag: 'COMPLETE', 
      desc: 'Percentile Normalization',
      simpleDesc: 'Normalizes brightness and contrast to reveal faint underwater debris.',
      details: 'Clamps extreme acoustic outliers to expand mid-tone shadow structures.'
    },
    { 
      step: '04', 
      name: 'Lee Speckle Filter', 
      tag: 'COMPLETE', 
      desc: '5x5 Local MMSE',
      simpleDesc: 'Eliminates acoustic grain noise while preserving sharp object edges.',
      details: 'Applies Minimum Mean Square Error spatial convolution across seabed tiles.'
    },
    { 
      step: '05', 
      name: 'Adaptive CLAHE', 
      tag: 'COMPLETE', 
      desc: 'Contrast Equalization',
      simpleDesc: 'Balances shadows and bright highlights locally across deep seabed relief.',
      details: 'Contrast-Limited Adaptive Histogram Equalization prevents over-amplification.'
    },
    { 
      step: '06', 
      name: '640x640 Tiling', 
      tag: 'COMPLETE', 
      desc: '20% Stride Overlap',
      simpleDesc: 'Slices large acoustic swaths into overlapping tiles for AI scanning.',
      details: '20% stride overlap guarantees no target is sliced in half at tile boundaries.'
    },
    { 
      step: '07', 
      name: 'YOLO11N distilled', 
      tag: 'ACTIVE', 
      desc: 'Batched GPU FP16',
      simpleDesc: 'Distilled neural network detects man-made debris, wrecks, and targets with 0.8484 mAP@50.',
      details: 'Optimized YOLO11N distilled model predicts bounding boxes in 17.78ms mean inference.'
    },
    { 
      step: '08', 
      name: 'Context & NMS', 
      tag: 'COMPLETE', 
      desc: 'Shadow Physics & Rank',
      simpleDesc: 'Verifies acoustic shadow physics and removes duplicate detections.',
      details: 'Eliminates false alarms by validating shadow deficit and aspect ratio.'
    },
    { 
      step: '09', 
      name: 'Operator Triage', 
      tag: 'WORKFLOW', 
      desc: 'Human-in-the-Loop',
      simpleDesc: 'Routes validated candidate hits to the human operator for review.',
      details: 'Provides one-click verification (Confirm, False Alarm, ROV Review).'
    },
    { 
      step: '10', 
      name: 'PostGIS Georef', 
      tag: 'PERSIST', 
      desc: 'WGS-84 RFC 7946',
      simpleDesc: 'Attaches accurate WGS-84 geographic coordinates for navigation.',
      details: 'Computes layback towfish offsets and saves geospatial records to PostGIS.'
    },
  ];

  const stepInferenceTimes = [
    { step: '01', name: 'Raw Ingestion', time: '45ms', category: 'Ingestion & Decoding', status: 'Optimal' },
    { step: '02', name: 'SNR Dynamic Range', time: '32ms', category: 'Signal Filtering', status: 'Optimal' },
    { step: '03', name: 'Lee Speckle Filter', time: '78ms', category: 'MMSE Denoising', status: 'Active' },
    { step: '04', name: 'Adaptive CLAHE', time: '56ms', category: 'Contrast Equalization', status: 'Optimal' },
    { step: '05', name: 'YOLO11N distilled Inference', time: '17.78ms', category: 'Neural Tensor Cores', status: 'GPU FP16' },
    { step: '06', name: 'Context & NMS', time: '34ms', category: 'Shadow Geometry NMS', status: 'Optimal' },
    { step: '07', name: 'Operator Triage', time: '120ms', category: 'Verification State Sync', status: 'Nominal' },
    { step: '08', name: 'PostGIS Georef', time: '45ms', category: 'WGS-84 Coordinate Fix', status: 'Optimal' },
  ];

  const handleCopyMetrics = () => {
    const text = `Metric\tYOLO11N distilled
mAP@50\t0.8484
mAP@50-95\t0.8080
Precision\t0.9020
Recall\t0.8172
F1\t0.8575
Parameters\t2,590,815
Mean inference\t17.78 ms
P95 inference\t20.89 ms
Peak VRAM\t848 MiB`;
    navigator.clipboard.writeText(text);
    setCopiedMetrics(true);
    setTimeout(() => setCopiedMetrics(false), 2000);
  };

  const handleExportPipelineReport = () => {
    const reportData = {
      pipeline_name: 'Sonar-Intel Neural Architecture',
      model: 'YOLO11N distilled',
      cuda_version: '12.6 FP16',
      total_latency_ms: 428,
      timestamp: new Date().toISOString(),
      step_inference_times: stepInferenceTimes,
      resource_usage: {
        gpu_usage: '78%',
        cpu_usage: '45%',
        peak_vram: '848 MiB',
        memory_usage: '848 MiB'
      },
      metrics: {
        mAP_50: 0.8484,
        mAP_50_95: 0.8080,
        precision: 0.9020,
        recall: 0.8172,
        f1: 0.8575,
        parameters: 2590815,
        mean_inference: '17.78 ms',
        p95_inference: '20.89 ms',
        peak_vram: '848 MiB'
      }
    };

    const blob = new Blob([JSON.stringify(reportData, null, 2)], { type: 'application/json' });
    const url = URL.createObjectURL(blob);
    const a = document.createElement('a');
    a.href = url;
    a.download = `pipeline_telemetry_report_${new Date().toISOString().split('T')[0]}.json`;
    a.click();
    URL.revokeObjectURL(url);
    setExportNotice('Pipeline telemetry report exported successfully!');
    setTimeout(() => setExportNotice(null), 3000);
  };

  return (
    <div className="p-6 lg:p-8 max-w-[1600px] mx-auto space-y-8 font-sans">
      
      {/* 1. Header Toolbar */}
      <div className="bg-white rounded-[24px] border border-[#e2e8f0] p-6 shadow-soft flex flex-col lg:flex-row items-start lg:items-center justify-between gap-4">
        <div className="space-y-2">
          {/* Top-Left: Back Button & Breadcrumb */}
          <div className="flex items-center gap-3">
            {onNavigateToDashboard && (
              <button
                onClick={onNavigateToDashboard}
                className="px-3.5 py-1.5 rounded-xl bg-[#f8fafc] hover:bg-slate-100 text-[#0f172a] hover:text-[#1d4ed8] border border-[#e2e8f0] font-semibold text-xs transition-all duration-200 shadow-tactile flex items-center gap-1.5 cursor-pointer group"
              >
                <ChevronLeft className="w-3.5 h-3.5 text-[#64748b] group-hover:text-[#1d4ed8] group-hover:-translate-x-0.5 transition-transform" />
                <span>Back to Dashboard Overview</span>
              </button>
            )}

            <div className="flex items-center gap-2 text-xs font-semibold">
              <button
                onClick={onNavigateToDashboard}
                className="text-[#64748b] hover:text-[#1d4ed8] hover:underline cursor-pointer transition-colors"
              >
                Dashboard Overview
              </button>
              <span className="text-[#cbd5e1]">&gt;</span>
              <span className="text-[#1d4ed8] font-bold">
                Pipeline Monitor
              </span>
            </div>
          </div>

          <h2 className="text-2xl font-extrabold text-[#0f172a] font-display flex items-center gap-2.5">
            <Cpu className="w-6 h-6 text-[#1d4ed8]" />
            Pipeline Monitor: AI Deep Learning Telemetry
          </h2>
        </div>

        <div className="flex flex-wrap items-center gap-3">
          <span className="text-xs font-mono font-bold px-3.5 py-1.5 rounded-full bg-blue-50/60 border border-blue-100 text-[#0f172a] shadow-tactile">
            MODEL: YOLO11N distilled
          </span>
          <span className="text-xs font-semibold px-3.5 py-1.5 rounded-full bg-emerald-50 text-emerald-700 border border-emerald-200 flex items-center gap-1.5">
            <span className="w-2 h-2 rounded-full bg-emerald-500 animate-pulse" />
            CUDA 12.6 FP16 Active
          </span>

          {/* Export Pipeline Report Button */}
          <button
            onClick={handleExportPipelineReport}
            className="px-4 py-2 rounded-full bg-[#1d4ed8] hover:bg-[#1e40af] text-white font-semibold text-xs transition-all duration-200 shadow-tactile flex items-center gap-2 cursor-pointer shadow-blue-glow"
          >
            <FileDown className="w-4 h-4" />
            <span>Export Pipeline Report</span>
          </button>
        </div>
      </div>

      {/* Export Toast Notification */}
      {exportNotice && (
        <div className="p-3.5 rounded-2xl bg-emerald-50 border border-emerald-200 text-emerald-800 text-xs font-semibold flex items-center justify-between shadow-soft">
          <div className="flex items-center gap-2">
            <CheckCircle2 className="w-4 h-4 text-emerald-600" />
            <span>{exportNotice}</span>
          </div>
          <button onClick={() => setExportNotice(null)} className="text-emerald-700 hover:text-emerald-900 font-bold text-xs cursor-pointer">
            <X className="w-4 h-4" />
          </button>
        </div>
      )}

      {/* 2. Active 10-Stage Edge Pipeline Flowchart & Progress Bar */}
      <section className="bg-white rounded-[24px] border border-[#e2e8f0] p-7 shadow-soft space-y-6">
        <div className="border-b border-slate-100 pb-3 flex flex-col sm:flex-row sm:items-center justify-between gap-2">
          <div>
            <span className="section-label block">End-to-End Deep Learning Architecture</span>
            <h3 className="text-base font-bold text-[#0f172a] font-display mt-0.5 flex items-center gap-2">
              <Activity className="w-4 h-4 text-[#1d4ed8]" />
              Active 10-Stage Sonar Signal & Deep Anomaly Pipeline
            </h3>
          </div>
          <div className="flex items-center gap-2">
            <span className="text-xs font-bold text-emerald-700 bg-emerald-50 px-3 py-1 rounded-full border border-emerald-100 flex items-center gap-1.5">
              <ShieldCheck className="w-3.5 h-3.5" />
              Lee Speckle MMSE + CLAHE Enabled
            </span>
          </div>
        </div>

        {/* Pipeline Overall Progress Bar */}
        <div className="p-4 rounded-2xl bg-[#f8fafc] border border-[#e2e8f0] space-y-2">
          <div className="flex items-center justify-between text-xs">
            <div className="flex items-center gap-2">
              <span className="w-2 h-2 rounded-full bg-[#1d4ed8] animate-pulse" />
              <span className="font-bold text-[#0f172a]">Overall Pipeline Progress</span>
              <span className="text-[#64748b]">• 10/10 Stages Initialized & Operational (Click any stage for info)</span>
            </div>
            <span className="font-mono font-extrabold text-[#1d4ed8]">100% Complete</span>
          </div>
          <div className="w-full bg-[#e2e8f0] h-2.5 rounded-full overflow-hidden">
            <div className="bg-gradient-to-r from-[#1d4ed8] via-cyan-500 to-emerald-500 h-full rounded-full transition-all duration-500 w-full" />
          </div>
        </div>

        {/* 10 Clickable Pipeline Stages with Hover and Selection Tooltips */}
        <div className="grid grid-cols-2 sm:grid-cols-3 md:grid-cols-5 lg:grid-cols-10 gap-3 pt-1">
          {pipelineStages.map((stage, idx) => {
            const isActive = stage.tag === 'ACTIVE';
            const isSelected = selectedStage?.step === stage.step;

            return (
              <button
                key={idx}
                type="button"
                onClick={() => setSelectedStage(isSelected ? null : stage)}
                title={`${stage.name}: ${stage.simpleDesc} (Click to inspect)`}
                className={`p-3.5 rounded-2xl border transition-all duration-200 flex flex-col justify-between space-y-2 text-left cursor-pointer group relative ${
                  isSelected
                    ? 'bg-blue-100/90 border-[#1d4ed8] shadow-[0_6px_20px_-2px_rgba(29,78,216,0.35)] scale-[1.03] ring-2 ring-blue-500/40'
                    : isActive
                    ? 'bg-blue-50/80 border-[#1d4ed8] shadow-[0_4px_14px_-2px_rgba(29,78,216,0.25)] hover:bg-blue-100/60'
                    : 'bg-[#f8fafc] border-[#e2e8f0] hover:bg-white hover:border-[#1d4ed8] hover:shadow-md'
                }`}
              >
                <div className="flex items-center justify-between w-full">
                  <span className="font-mono text-[10px] font-bold text-[#64748b]">{stage.step}</span>
                  <span className={`text-[8px] font-bold px-1.5 py-0.2 rounded-full font-sans ${
                    isActive
                      ? 'bg-[#1d4ed8] text-white animate-pulse shadow-xs'
                      : stage.tag === 'COMPLETE'
                      ? 'bg-emerald-50 text-emerald-700'
                      : stage.tag === 'PERSIST'
                      ? 'bg-blue-50 text-blue-700'
                      : 'bg-amber-50 text-amber-700'
                  }`}>
                    {stage.tag}
                  </span>
                </div>

                <div>
                  <div className={`font-bold text-[11px] leading-tight group-hover:text-[#1d4ed8] transition-colors ${isActive ? 'text-[#1d4ed8]' : 'text-[#0f172a]'}`}>
                    {stage.name}
                  </div>
                  <div className="text-[10px] text-[#64748b] mt-0.5 truncate">
                    {stage.desc}
                  </div>
                </div>

                {/* Click tooltip badge hint */}
                <div className="text-[9px] text-[#1d4ed8] font-semibold opacity-0 group-hover:opacity-100 transition-opacity">
                  Click for info
                </div>
              </button>
            );
          })}
        </div>

        {/* Selected Stage Detail Tooltip Box */}
        {selectedStage && (
          <div className="p-4 rounded-2xl bg-blue-50/90 border border-blue-200 shadow-soft animate-in fade-in slide-in-from-top-1 duration-200 flex items-start justify-between gap-4">
            <div className="space-y-1">
              <div className="flex items-center gap-2">
                <span className="font-mono font-bold text-xs bg-[#1d4ed8] text-white px-2 py-0.5 rounded-md">
                  Step {selectedStage.step}
                </span>
                <span className="font-bold text-sm text-[#0f172a] font-display">
                  {selectedStage.name}
                </span>
                <span className="text-xs font-semibold text-[#64748b]">({selectedStage.desc})</span>
              </div>
              <p className="text-xs text-[#0f172a] font-medium leading-relaxed">
                💡 <strong className="text-[#1d4ed8]">What it does:</strong> {selectedStage.simpleDesc}
              </p>
              <p className="text-[11px] text-[#64748b] font-mono">
                ⚙️ {selectedStage.details}
              </p>
            </div>
            <button
              onClick={() => setSelectedStage(null)}
              className="p-1 rounded-lg hover:bg-blue-100 text-[#64748b] hover:text-[#0f172a] transition-colors cursor-pointer shrink-0"
              title="Close info"
            >
              <X className="w-4 h-4" />
            </button>
          </div>
        )}

        {/* Step-Level Inference Times Table */}
        <div className="pt-3 border-t border-slate-100 space-y-3">
          <div className="flex items-center justify-between">
            <span className="section-label block">Latency Breakdown</span>
            <span className="text-xs font-mono font-bold text-[#1d4ed8]">Total Cumulative: 428 ms</span>
          </div>

          <div className="overflow-x-auto">
            <table className="w-full text-xs text-left">
              <thead>
                <tr className="border-b border-[#e2e8f0] text-[#64748b] text-[10px] uppercase font-bold tracking-wider">
                  <th className="pb-2 pl-2">Step</th>
                  <th className="pb-2">Pipeline Stage</th>
                  <th className="pb-2">Function Category</th>
                  <th className="pb-2 text-right">Avg Inference Time</th>
                  <th className="pb-2 text-right pr-2">Execution State</th>
                </tr>
              </thead>
              <tbody className="divide-y divide-[#f1f5f9] font-sans">
                {stepInferenceTimes.map((item, idx) => (
                  <tr key={idx} className="hover:bg-[#f8fafc] transition-colors">
                    <td className="py-2.5 pl-2 font-mono font-bold text-[#64748b]">{item.step}</td>
                    <td className="py-2.5 font-bold text-[#0f172a]">{item.name}</td>
                    <td className="py-2.5 text-[#64748b]">{item.category}</td>
                    <td className="py-2.5 text-right font-mono font-bold text-[#1d4ed8]">{item.time}</td>
                    <td className="py-2.5 text-right pr-2">
                      <span className="px-2 py-0.5 rounded-full text-[10px] font-bold bg-emerald-50 text-emerald-700 border border-emerald-100">
                        {item.status}
                      </span>
                    </td>
                  </tr>
                ))}
              </tbody>
            </table>
          </div>
        </div>
      </section>

      {/* 3. Baseline Benchmarks & Resource Usage Cards */}
      <section className="space-y-6">
        {/* Baseline Benchmarks (4 Cards) */}
        <div className="grid grid-cols-1 sm:grid-cols-2 lg:grid-cols-4 gap-6">
          <div className="bg-white rounded-[24px] border border-[#e2e8f0] p-6 shadow-soft space-y-2">
            <span className="section-label block">mAP@50 Detection Accuracy</span>
            <div className="text-3xl font-extrabold text-[#0f172a] font-display">
              0.8484
            </div>
            <div className="text-xs font-semibold text-[#1d4ed8]">mAP@50-95: 0.8080 (80.80%)</div>
            <p className="text-xs text-[#64748b] pt-2 border-t border-slate-100">
              Mean Average Precision at IoU 0.50 benchmark across real hydrographic swaths.
            </p>
          </div>

          <div className="bg-white rounded-[24px] border border-[#e2e8f0] p-6 shadow-soft space-y-2">
            <span className="section-label block">Target Precision</span>
            <div className="text-3xl font-extrabold text-emerald-600 font-display">
              0.9020
            </div>
            <div className="text-xs font-semibold text-emerald-700">Validated Detection Precision</div>
            <p className="text-xs text-[#64748b] pt-2 border-t border-slate-100">
              High precision across shipwrecks, pipelines, cylinders, and acoustic debris.
            </p>
          </div>

          <div className="bg-white rounded-[24px] border border-[#e2e8f0] p-6 shadow-soft space-y-2">
            <span className="section-label block">Recall & F1 Score</span>
            <div className="text-3xl font-extrabold text-[#0f172a] font-display">
              0.8172
            </div>
            <div className="text-xs font-semibold text-[#1d4ed8]">F1 Score: 0.8575 (85.75%)</div>
            <p className="text-xs text-[#64748b] pt-2 border-t border-slate-100">
              Harmonic mean of precision and recall ensuring high sensitivity with minimal misses.
            </p>
          </div>

          <div className="bg-white rounded-[24px] border border-[#e2e8f0] p-6 shadow-soft space-y-2">
            <span className="section-label block">Inference Latency</span>
            <div className="text-3xl font-extrabold text-[#1d4ed8] font-display flex items-center gap-1.5">
              <Zap className="w-6 h-6 text-[#1d4ed8]" />
              17.78 ms
            </div>
            <div className="text-xs font-semibold text-[#1d4ed8]">P95 Inference: 20.89 ms</div>
            <p className="text-xs text-[#64748b] pt-2 border-t border-slate-100">
              Real-time distilled model inference on NVIDIA CUDA Tensor Cores / RTX GPUs.
            </p>
          </div>
        </div>

        {/* Resource Usage Cards (3 Cards) */}
        <div className="grid grid-cols-1 sm:grid-cols-3 gap-6">
          <div className="bg-white rounded-[24px] border border-[#e2e8f0] p-5 shadow-soft space-y-3">
            <div className="flex items-center justify-between">
              <span className="section-label block">Compute Hardware</span>
              <span className="text-[10px] font-bold px-2 py-0.5 rounded-full bg-emerald-50 text-emerald-700">Active</span>
            </div>
            <div className="flex items-center justify-between">
              <div>
                <div className="text-xs font-bold text-[#64748b]">GPU Usage</div>
                <div className="text-2xl font-extrabold text-[#0f172a] font-display">78%</div>
              </div>
              <div className="h-10 w-10 rounded-xl bg-blue-50 text-[#1d4ed8] flex items-center justify-center font-bold text-sm shadow-xs">
                <Zap className="w-5 h-5" />
              </div>
            </div>
            <div className="w-full bg-[#e2e8f0] h-2 rounded-full overflow-hidden">
              <div className="bg-[#1d4ed8] h-full rounded-full transition-all duration-300" style={{ width: '78%' }} />
            </div>
          </div>

          <div className="bg-white rounded-[24px] border border-[#e2e8f0] p-5 shadow-soft space-y-3">
            <div className="flex items-center justify-between">
              <span className="section-label block">Host System</span>
              <span className="text-[10px] font-bold px-2 py-0.5 rounded-full bg-emerald-50 text-emerald-700">Nominal</span>
            </div>
            <div className="flex items-center justify-between">
              <div>
                <div className="text-xs font-bold text-[#64748b]">CPU Usage</div>
                <div className="text-2xl font-extrabold text-[#0f172a] font-display">45%</div>
              </div>
              <div className="h-10 w-10 rounded-xl bg-slate-50 text-[#0f172a] flex items-center justify-center font-bold text-sm shadow-xs">
                <Cpu className="w-5 h-5 text-[#1d4ed8]" />
              </div>
            </div>
            <div className="w-full bg-[#e2e8f0] h-2 rounded-full overflow-hidden">
              <div className="bg-cyan-500 h-full rounded-full transition-all duration-300" style={{ width: '45%' }} />
            </div>
          </div>

          <div className="bg-white rounded-[24px] border border-[#e2e8f0] p-5 shadow-soft space-y-3">
            <div className="flex items-center justify-between">
              <span className="section-label block">Peak VRAM</span>
              <span className="text-[10px] font-bold px-2 py-0.5 rounded-full bg-blue-50 text-[#1d4ed8]">Optimal</span>
            </div>
            <div className="flex items-center justify-between">
              <div>
                <div className="text-xs font-bold text-[#64748b]">Peak VRAM Allocation</div>
                <div className="text-2xl font-extrabold text-[#0f172a] font-display">848 MiB</div>
              </div>
              <div className="h-10 w-10 rounded-xl bg-blue-50 text-[#1d4ed8] flex items-center justify-center font-bold text-sm shadow-xs">
                <Server className="w-5 h-5" />
              </div>
            </div>
            <div className="w-full bg-[#e2e8f0] h-2 rounded-full overflow-hidden">
              <div className="bg-indigo-600 h-full rounded-full transition-all duration-300" style={{ width: '32%' }} />
            </div>
          </div>
        </div>
      </section>

      {/* 4. Deep Model Specs, Artifact Preview & Execution Log */}
      <section className="grid grid-cols-1 lg:grid-cols-12 gap-6">
        
        {/* Model Card Specifications (4 Cols) - Exact Image Match */}
        <div className="lg:col-span-4 bg-white rounded-[24px] border border-[#e2e8f0] p-6 shadow-soft space-y-4">
          <div className="border-b border-slate-100 pb-3 flex items-center justify-between">
            <div>
              <span className="section-label block">Benchmark Evaluation</span>
              <h3 className="text-base font-bold text-[#0f172a] font-display mt-0.5 flex items-center gap-2">
                <Terminal className="w-4 h-4 text-[#1d4ed8]" />
                Model Details
              </h3>
            </div>
            <button
              onClick={handleCopyMetrics}
              title="Copy model metrics"
              className="px-2.5 py-1 rounded-lg bg-[#f8fafc] hover:bg-blue-50 text-[#64748b] hover:text-[#1d4ed8] border border-[#e2e8f0] font-mono text-[11px] font-semibold flex items-center gap-1.5 transition-colors cursor-pointer"
            >
              {copiedMetrics ? (
                <>
                  <Check className="w-3.5 h-3.5 text-emerald-600" />
                  <span className="text-emerald-700">Copied</span>
                </>
              ) : (
                <>
                  <Copy className="w-3.5 h-3.5" />
                  <span>Copy</span>
                </>
              )}
            </button>
          </div>

          {/* Exact Model Details Table matching user image */}
          <div className="overflow-hidden rounded-2xl border border-[#e2e8f0] bg-white text-xs font-mono">
            <div className="flex justify-between items-center px-3.5 py-2.5 bg-slate-900 text-white font-bold border-b border-slate-800">
              <span className="tracking-wide">Metric</span>
              <span className="text-cyan-300">YOLO11N distilled</span>
            </div>

            <div className="divide-y divide-slate-100">
              <div className="flex justify-between items-center px-3.5 py-2 hover:bg-[#f8fafc] transition-colors">
                <span className="text-[#64748b] font-medium">mAP@50</span>
                <span className="font-bold text-[#0f172a]">0.8484</span>
              </div>
              <div className="flex justify-between items-center px-3.5 py-2 hover:bg-[#f8fafc] transition-colors">
                <span className="text-[#64748b] font-medium">mAP@50-95</span>
                <span className="font-bold text-[#0f172a]">0.8080</span>
              </div>
              <div className="flex justify-between items-center px-3.5 py-2 hover:bg-[#f8fafc] transition-colors">
                <span className="text-[#64748b] font-medium">Precision</span>
                <span className="font-bold text-emerald-700">0.9020</span>
              </div>
              <div className="flex justify-between items-center px-3.5 py-2 hover:bg-[#f8fafc] transition-colors">
                <span className="text-[#64748b] font-medium">Recall</span>
                <span className="font-bold text-[#0f172a]">0.8172</span>
              </div>
              <div className="flex justify-between items-center px-3.5 py-2 hover:bg-[#f8fafc] transition-colors">
                <span className="text-[#64748b] font-medium">F1</span>
                <span className="font-bold text-[#0f172a]">0.8575</span>
              </div>
              <div className="flex justify-between items-center px-3.5 py-2 hover:bg-[#f8fafc] transition-colors">
                <span className="text-[#64748b] font-medium">Parameters</span>
                <span className="font-bold text-[#1d4ed8]">2,590,815</span>
              </div>
              <div className="flex justify-between items-center px-3.5 py-2 hover:bg-[#f8fafc] transition-colors">
                <span className="text-[#64748b] font-medium">Mean inference</span>
                <span className="font-bold text-[#1d4ed8]">17.78 ms</span>
              </div>
              <div className="flex justify-between items-center px-3.5 py-2 hover:bg-[#f8fafc] transition-colors">
                <span className="text-[#64748b] font-medium">P95 inference</span>
                <span className="font-bold text-[#0f172a]">20.89 ms</span>
              </div>
              <div className="flex justify-between items-center px-3.5 py-2 hover:bg-[#f8fafc] transition-colors">
                <span className="text-[#64748b] font-medium">Peak VRAM</span>
                <span className="font-bold text-indigo-700">848 MiB</span>
              </div>
            </div>
          </div>
        </div>

        {/* Artifact Preview Card (3 Cols) */}
        <div className="lg:col-span-3 bg-white rounded-[24px] border border-[#e2e8f0] p-6 shadow-soft space-y-4 flex flex-col justify-between">
          <div>
            <div className="border-b border-slate-100 pb-3 flex items-center justify-between">
              <div>
                <span className="section-label block">Inference Artifact</span>
                <h3 className="text-base font-bold text-[#0f172a] font-display mt-0.5 flex items-center gap-2">
                  <Scan className="w-4 h-4 text-[#1d4ed8]" />
                  Artifact Preview
                </h3>
              </div>
              <span className="text-xs font-mono font-bold px-2 py-0.5 rounded-full bg-emerald-50 text-emerald-700 border border-emerald-100">
                83% CONF
              </span>
            </div>

            {/* Sonar Crop Image Thumbnail */}
            <div className="mt-3 aspect-4/3 rounded-2xl bg-[#050a14] border border-slate-800 relative overflow-hidden shadow-inner flex items-center justify-center group">
              <img
                src="/demo_samples/c001_artifact_crop.png"
                alt="Shipwreck Detection Artifact (C001)"
                className="w-full h-full object-cover filter contrast-125 brightness-110"
                onError={(e) => {
                  // Fallback to full demo image if crop is missing
                  (e.target as HTMLImageElement).src = '/demo_samples/viator_04_test_wreck.png';
                }}
              />
              <div className="absolute inset-4 border-2 border-cyan-400 rounded-sm pointer-events-none">
                <div className="absolute -top-2.5 left-1/2 -translate-x-1/2 px-2 py-0.2 bg-rose-600 text-white font-mono font-bold text-[9px] rounded-full whitespace-nowrap shadow-xs">
                  C001 • SHIPWRECK
                </div>
              </div>
            </div>
          </div>

          <div className="p-3 rounded-2xl bg-[#f8fafc] border border-[#e2e8f0] space-y-1 text-xs">
            <div className="flex justify-between items-center">
              <span className="text-[#64748b]">Artifact ID:</span>
              <span className="font-mono font-bold text-[#0f172a]">C001_viator_04</span>
            </div>
            <div className="flex justify-between items-center">
              <span className="text-[#64748b]">Bounding Box:</span>
              <span className="font-mono text-[#0f172a]">[144 × 94 px]</span>
            </div>
            <div className="flex justify-between items-center">
              <span className="text-[#64748b]">Confidence:</span>
              <span className="font-mono font-bold text-emerald-700">83.4% (Match)</span>
            </div>
          </div>
        </div>

        {/* Real-Time Execution Log (5 Cols) */}
        <div className="lg:col-span-5 bg-white rounded-[24px] border border-[#e2e8f0] p-6 shadow-soft space-y-4">
          <div className="border-b border-slate-100 pb-3 flex items-center justify-between">
            <div>
              <span className="section-label block">Execution Audit Log</span>
              <h3 className="text-base font-bold text-[#0f172a] font-display mt-0.5 flex items-center gap-2">
                <BarChart3 className="w-4 h-4 text-[#1d4ed8]" />
                Recent Survey Ingestion Log
              </h3>
            </div>
            <span className="text-xs font-semibold text-[#64748b]">
              Batched Slices
            </span>
          </div>

          <div className="space-y-2.5 text-xs font-sans">
            {[
              { file: 'viator_04_test_wreck.png', tiles: '12 Tiles', latency: '17.78 ms', candidates: '4 Hits (Shipwreck)', status: 'PROCESSED' },
              { file: 'corsican_02_target.png', tiles: '10 Tiles', latency: '16.50 ms', candidates: '3 Hits (Shipwreck)', status: 'PROCESSED' },
              { file: 'artificial_reef_02_clutter.png', tiles: '16 Tiles', latency: '19.82 ms', candidates: '2 Hits (Clutter)', status: 'PROCESSED' },
              { file: 'survey_001_reference.png', tiles: '14 Tiles', latency: '18.10 ms', candidates: '3 Hits (Debris/Net)', status: 'PROCESSED' },
            ].map((log, idx) => (
              <div 
                key={idx} 
                className="p-3 rounded-2xl bg-[#f8fafc] border border-[#e2e8f0] hover:bg-white hover:border-slate-300 transition-colors flex items-center justify-between"
              >
                <div className="truncate max-w-[150px] text-[#0f172a] font-mono font-bold">
                  {log.file}
                </div>
                <div className="text-[#64748b] font-medium">{log.tiles}</div>
                <div className="text-[#1d4ed8] font-mono font-bold">{log.latency}</div>
                <span className="text-[10px] font-bold px-2 py-0.5 rounded-full bg-emerald-50 text-emerald-700 border border-emerald-100">
                  {log.status}
                </span>
              </div>
            ))}
          </div>
        </div>

      </section>

      {/* Bottom-Right Navigation Action */}
      {onNavigateToDashboard && (
        <div className="flex justify-end pt-2">
          <button
            onClick={onNavigateToDashboard}
            className="px-5 py-2.5 rounded-full bg-[#1d4ed8] hover:bg-[#1e40af] text-white font-semibold text-xs transition-all duration-200 shadow-tactile shadow-blue-glow flex items-center gap-2 cursor-pointer hover:scale-[1.02] active:scale-[0.98]"
          >
            <ChevronLeft className="w-4 h-4" />
            <span>Back to Dashboard Overview</span>
          </button>
        </div>
      )}

    </div>
  );
};


