import React from 'react';
import { Contact } from '../../types/detection';

interface DetectionOverlayProps {
  contacts: Contact[];
  selectedContact: Contact | null;
  onSelectContact: (contact: Contact) => void;
  scaleX: number;
  scaleY: number;
}

export const DetectionOverlay: React.FC<DetectionOverlayProps> = ({
  contacts,
  selectedContact,
  onSelectContact,
  scaleX,
  scaleY
}) => {
  return (
    <svg className="absolute inset-0 w-full h-full pointer-events-none z-10">
      {contacts.map((contact) => {
        const isSelected = selectedContact?.contact_id === contact.contact_id;
        const x = contact.bbox.x1 * scaleX;
        const y = contact.bbox.y1 * scaleY;
        const width = (contact.bbox.x2 - contact.bbox.x1) * scaleX;
        const height = (contact.bbox.y2 - contact.bbox.y1) * scaleY;

        let strokeColor = '#38bdf8'; // LOW
        let fillColor = 'rgba(56, 189, 248, 0.15)';
        if (contact.priority === 'HIGH') {
          strokeColor = '#ef4444';
          fillColor = 'rgba(239, 68, 68, 0.20)';
        } else if (contact.priority === 'MEDIUM') {
          strokeColor = '#f59e0b';
          fillColor = 'rgba(245, 158, 11, 0.20)';
        }

        return (
          <g
            key={contact.contact_id}
            className="cursor-pointer pointer-events-auto"
            onClick={(e) => {
              e.stopPropagation();
              onSelectContact(contact);
            }}
          >
            {/* Bounding Box rectangle */}
            <rect
              x={x}
              y={y}
              width={width}
              height={height}
              fill={fillColor}
              stroke={strokeColor}
              strokeWidth={isSelected ? 3 : 1.5}
              strokeDasharray={isSelected ? 'none' : '4 2'}
              className="transition-all hover:fill-opacity-40"
            />

            {/* Corner brackets when selected */}
            {isSelected && (
              <>
                <polyline
                  points={`${x - 4},${y + 8} ${x - 4},${y - 4} ${x + 8},${y - 4}`}
                  fill="none"
                  stroke="#00f0ff"
                  strokeWidth="2.5"
                />
                <polyline
                  points={`${x + width - 8},${y - 4} ${x + width + 4},${y - 4} ${x + width + 4},${y + 8}`}
                  fill="none"
                  stroke="#00f0ff"
                  strokeWidth="2.5"
                />
                <polyline
                  points={`${x - 4},${y + height - 8} ${x - 4},${y + height + 4} ${x + 8},${y + height + 4}`}
                  fill="none"
                  stroke="#00f0ff"
                  strokeWidth="2.5"
                />
                <polyline
                  points={`${x + width - 8},${y + height + 4} ${x + width + 4},${y + height + 4} ${x + width + 4},${y + height - 8}`}
                  fill="none"
                  stroke="#00f0ff"
                  strokeWidth="2.5"
                />
              </>
            )}

            {/* ID & Priority Label Tag */}
            <g transform={`translate(${x}, ${Math.max(16, y - 6)})`}>
              <rect
                x="-1"
                y="-14"
                width={78}
                height="15"
                fill="#070e1a"
                stroke={strokeColor}
                strokeWidth="1"
                rx="2"
              />
              <text
                x="3"
                y="-3"
                fill={strokeColor}
                fontSize="10"
                fontFamily="JetBrains Mono, monospace"
                fontWeight="bold"
              >
                {contact.contact_id} · {contact.priority[0]}
              </text>
            </g>
          </g>
        );
      })}
    </svg>
  );
};
