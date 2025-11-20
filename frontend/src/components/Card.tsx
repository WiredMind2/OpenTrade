import React from 'react'

export default function Card({ children }: { children: React.ReactNode }) {
  return <div className="bg-white p-4 shadow rounded">{children}</div>
}
