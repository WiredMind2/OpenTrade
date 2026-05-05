import { Navigate, useLocation } from 'react-router-dom'

/** Legacy `/trade-plan` URLs redirect to Predictions (plan builder lives there). Query params are preserved. */
export default function TradePlanRedirect() {
  const { search } = useLocation()
  return <Navigate to={{ pathname: '/predictions', search }} replace />
}
