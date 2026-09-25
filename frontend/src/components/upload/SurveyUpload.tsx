import React, { useState, useRef } from 'react';
import { UploadCloud, FileText, Image as ImageIcon, AlertCircle, CheckCircle, Sparkles } from 'lucide-react';
import { apiService } from '../../services/api';
import { SurveyUploadResponse } from '../../types/detection';

interface SurveyUploadProps {
  onUploadSuccess: (survey: SurveyUploadResponse) => void;
}

export const SurveyUpload: React.FC<SurveyUploadProps> = ({ onUploadSuccess }) => {
  const [sonarFile, setSonarFile] = useState<File | null>(null);
  const [navFile, setNavFile] = useState<File | null>(null);
  const [uploading, setUploading] = useState<boolean>(false);
  const [error, setError] = useState<string | null>(null);
  const [dragActive, setDragActive] = useState<boolean>(false);

  const sonarInputRef = useRef<HTMLInputElement>(null);
  const navInputRef = useRef<HTMLInputElement>(null);

  const handleDrag = (e: React.DragEvent) => {
    e.preventDefault();
    e.stopPropagation();
    if (e.type === 'dragenter' || e.type === 'dragover') {
      setDragActive(true);
    } else if (e.type === 'dragleave') {
      setDragActive(false);
    }
  };

  const handleDrop = (e: React.DragEvent) => {
    e.preventDefault();
    e.stopPropagation();
    setDragActive(false);
    if (e.dataTransfer.files && e.dataTransfer.files[0]) {
      const file = e.dataTransfer.files[0];
      if (file.type.startsWith('image/')) {
        setSonarFile(file);
      } else if (file.name.endsWith('.csv')) {
        setNavFile(file);
      }
    }
  };

  const handleSubmit = async (e: React.FormEvent) => {
    e.preventDefault();
    if (!sonarFile) {
      setError('Please select a Side-Scan Sonar (SSS) image file.');
      return;
    }
    setUploading(true);
    setError(null);
    try {
      const res = await apiService.uploadSurvey(sonarFile, navFile || undefined);
      onUploadSuccess(res);
    } catch (err: any) {
      setError(err.response?.data?.detail || 'Failed to ingest survey. Please verify file format.');
    } finally {
      setUploading(false);
    }
  };

  return (
    <div className="max-w-xl mx-auto p-6 bg-[#0a1424] border border-[#1a2f4c] rounded-lg shadow-2xl">
      <div className="flex items-center gap-2 mb-4">
        <UploadCloud className="w-5 h-5 text-cyan-400" />
        <h3 className="text-sm font-bold font-mono tracking-wider text-slate-100 uppercase">
          Ingest Mission Survey Swath
        </h3>
      </div>

      <form onSubmit={handleSubmit} className="space-y-4">
        {/* Sonar Image Dropzone */}
        <div
          onDragEnter={handleDrag}
          onDragLeave={handleDrag}
          onDragOver={handleDrag}
          onDrop={handleDrop}
          onClick={() => sonarInputRef.current?.click()}
          className={`border-2 border-dashed rounded-lg p-6 text-center cursor-pointer transition-all ${
            dragActive
              ? 'border-cyan-400 bg-cyan-950/20'
              : sonarFile
              ? 'border-emerald-600/60 bg-emerald-950/10'
              : 'border-[#1e385c] hover:border-cyan-600 bg-[#070e1a]'
          }`}
        >
          <input
            ref={sonarInputRef}
            type="file"
            accept="image/png,image/jpeg,image/tiff"
            className="hidden"
            onChange={(e) => e.target.files?.[0] && setSonarFile(e.target.files[0])}
          />
          <div className="flex flex-col items-center gap-2">
            <ImageIcon className={`w-8 h-8 ${sonarFile ? 'text-emerald-400' : 'text-slate-500'}`} />
            <div className="text-xs font-mono">
              {sonarFile ? (
                <span className="text-emerald-300 font-semibold">{sonarFile.name} ({(sonarFile.size / (1024 * 1024)).toFixed(2)} MB)</span>
              ) : (
                <span className="text-slate-300">Drop raw SSS waterfall image here or <span className="text-cyan-400 underline">browse</span></span>
              )}
            </div>
            <p className="text-[11px] text-slate-500">Supports PNG, JPG, or TIFF acoustic backscatter swaths</p>
          </div>
        </div>

        {/* Optional Navigation Track CSV */}
        <div
          onClick={() => navInputRef.current?.click()}
          className="border border-[#1a2f4c] hover:border-cyan-700 rounded p-3 bg-[#070e1a] cursor-pointer flex items-center justify-between transition-colors"
        >
          <input
            ref={navInputRef}
            type="file"
            accept=".csv"
            className="hidden"
            onChange={(e) => e.target.files?.[0] && setNavFile(e.target.files[0])}
          />
          <div className="flex items-center gap-2 text-xs font-mono">
            <FileText className={`w-4 h-4 ${navFile ? 'text-cyan-400' : 'text-slate-500'}`} />
            {navFile ? (
              <span className="text-cyan-300">{navFile.name}</span>
            ) : (
              <span className="text-slate-400">Attach Navigation Sensor CSV <span className="text-slate-600">(optional for WGS84 estimation)</span></span>
            )}
          </div>
          {navFile && (
            <span className="text-[10px] text-cyan-400 border border-cyan-800 px-1.5 py-0.5 rounded font-mono">
              READY
            </span>
          )}
        </div>

        {error && (
          <div className="p-2.5 rounded bg-red-950/40 border border-red-800 text-red-300 text-xs flex items-center gap-2 font-mono">
            <AlertCircle className="w-4 h-4 shrink-0" />
            <span>{error}</span>
          </div>
        )}

        <button
          type="submit"
          disabled={!sonarFile || uploading}
          className="w-full py-2.5 rounded font-mono text-xs font-bold text-slate-950 bg-cyan-400 hover:bg-cyan-300 disabled:opacity-50 disabled:hover:bg-cyan-400 transition-all shadow-[0_0_12px_rgba(6,182,212,0.25)] flex items-center justify-center gap-2"
        >
          {uploading ? 'VALIDATING & INGESTING...' : 'UPLOAD & INITIALIZE MISSION'}
        </button>
      </form>
    </div>
  );
};
