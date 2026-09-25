import React, { useEffect, useRef, useState } from 'react';
import maplibregl from 'maplibre-gl';
import 'maplibre-gl/dist/maplibre-gl.css';
import { Contact, NavWaypoint } from '../../types/detection';
import { MapPin, Navigation, Compass, Layers, Radio } from 'lucide-react';

interface MapViewProps {
  contacts: Contact[];
  selectedContact: Contact | null;
  navTrack: NavWaypoint[];
  onSelectContact: (contact: Contact) => void;
}

export const MapView: React.FC<MapViewProps> = ({
  contacts,
  selectedContact,
  navTrack,
  onSelectContact
}) => {
  const mapContainer = useRef<HTMLDivElement>(null);
  const mapInstance = useRef<maplibregl.Map | null>(null);
  const markersRef = useRef<{ [key: string]: maplibregl.Marker }>({});
  const [mapReady, setMapReady] = useState<boolean>(false);

  // Default coordinate center (Coastal India test coordinates)
  const defaultCenter: [number, number] = [76.5435, 11.2348];

  useEffect(() => {
    if (!mapContainer.current || mapInstance.current) return;

    try {
      const map = new maplibregl.Map({
        container: mapContainer.current,
        style: {
          version: 8,
          sources: {
            'dark-canvas': {
              type: 'raster',
              tiles: [
                'https://services.arcgisonline.com/arcgis/rest/services/Canvas/World_Dark_Gray_Base/MapServer/tile/{z}/{y}/{x}'
              ],
              tileSize: 256,
              maxzoom: 16,
              attribution: '© Esri, HERE, Garmin'
            }
          },
          layers: [
            {
              id: 'dark-canvas-layer',
              type: 'raster',
              source: 'dark-canvas',
              minzoom: 0,
              maxzoom: 22
            }
          ]
        },
        center: defaultCenter,
        zoom: 15,
        attributionControl: false
      });

      map.addControl(new maplibregl.NavigationControl({ showCompass: true }), 'top-right');

      map.on('load', () => {
        mapInstance.current = map;
        setMapReady(true);
      });

      return () => {
        map.remove();
        mapInstance.current = null;
      };
    } catch (err) {
      console.warn('MapLibre init error:', err);
    }
  }, []);

  // Update Markers
  useEffect(() => {
    if (!mapInstance.current || !mapReady) return;
    const map = mapInstance.current;

    // Clear old markers
    Object.values(markersRef.current).forEach(m => m.remove());
    markersRef.current = {};

    contacts.forEach(contact => {
      if (contact.latitude == null || contact.longitude == null) return;

      const isSelected = selectedContact?.contact_id === contact.contact_id;
      const el = document.createElement('div');
      el.className = 'cursor-pointer select-none transition-transform hover:scale-125';

      let bg = '#06b6d4';
      let glow = '0 0 12px rgba(6, 182, 212, 0.8)';
      if (contact.priority === 'HIGH') {
        bg = '#ef4444';
        glow = '0 0 14px rgba(239, 68, 68, 0.9)';
      } else if (contact.priority === 'MEDIUM') {
        bg = '#f59e0b';
        glow = '0 0 12px rgba(245, 158, 11, 0.8)';
      }

      el.innerHTML = `
        <div style="
          background-color: ${bg};
          width: ${isSelected ? '24px' : '18px'};
          height: ${isSelected ? '24px' : '18px'};
          border-radius: 50%;
          border: 2px solid #ffffff;
          box-shadow: ${glow};
          display: flex;
          align-items: center;
          justify-content: center;
          font-family: monospace;
          font-size: 9px;
          font-weight: 800;
          color: #000;
        ">
          ${contact.contact_id}
        </div>
      `;

      el.addEventListener('click', () => {
        onSelectContact(contact);
      });

      const marker = new maplibregl.Marker({ element: el })
        .setLngLat([contact.longitude, contact.latitude])
        .addTo(map);

      markersRef.current[contact.contact_id] = marker;
    });
  }, [contacts, selectedContact, mapReady, onSelectContact]);

  // Center map on selected contact
  useEffect(() => {
    if (!mapInstance.current || !selectedContact) return;
    if (selectedContact.longitude != null && selectedContact.latitude != null) {
      mapInstance.current.flyTo({
        center: [selectedContact.longitude, selectedContact.latitude],
        zoom: 16,
        essential: true
      });
    }
  }, [selectedContact]);

  return (
    <div className="relative flex-1 h-full w-full bg-[#030712] overflow-hidden select-none">
      {/* MapLibre DOM container */}
      <div ref={mapContainer} className="w-full h-full" />

      {/* Map Legend Overlay */}
      <div className="absolute bottom-4 left-4 p-3 rounded-xl bg-[#070e1e]/90 backdrop-blur-md border border-[#14244a] text-[11px] font-mono space-y-1.5 z-10 shadow-lg">
        <div className="text-[10px] text-cyan-400 font-bold uppercase tracking-wider mb-1 flex items-center gap-1.5">
          <Radio className="w-3 h-3 text-cyan-400" /> SPATIAL TARGET CLASSIFICATION
        </div>
        <div className="flex items-center gap-2">
          <div className="w-2.5 h-2.5 rounded-full bg-red-500 shadow-[0_0_8px_#ef4444]" />
          <span className="text-slate-200">HIGH PRIORITY (Deficit Shadow)</span>
        </div>
        <div className="flex items-center gap-2">
          <div className="w-2.5 h-2.5 rounded-full bg-amber-500 shadow-[0_0_8px_#f59e0b]" />
          <span className="text-slate-200">MEDIUM PRIORITY (Proposal)</span>
        </div>
        <div className="flex items-center gap-2">
          <div className="w-2.5 h-2.5 rounded-full bg-cyan-400 shadow-[0_0_8px_#06b6d4]" />
          <span className="text-slate-200">LOW / UNVERIFIED</span>
        </div>
      </div>

      {/* Coordinate & Heading Telemetry */}
      {selectedContact && selectedContact.latitude != null && (
        <div className="absolute top-4 left-4 p-2.5 rounded-xl bg-[#070e1e]/95 backdrop-blur-md border border-[#172b54] text-xs font-mono text-cyan-300 flex items-center gap-2.5 z-10 shadow-lg">
          <Navigation className="w-4 h-4 text-cyan-400" />
          <span>
            {selectedContact.latitude.toFixed(6)}° N, {selectedContact.longitude?.toFixed(6)}° E
          </span>
          <span className="text-[10px] px-2 py-0.5 rounded-full bg-cyan-950 border border-cyan-800 text-cyan-300 font-bold">
            {selectedContact.localization_status}
          </span>
        </div>
      )}
    </div>
  );
};
