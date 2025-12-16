import { useState, useEffect } from 'react'
import { listStrategies } from '../api/strategies'
import { StrategyMetadata } from '../types'
import Loading from '../components/Loading'
import ErrorMessage from '../components/ErrorMessage'
import StrategyCard from '../components/StrategyCard'

function Strategies() {
  const [strategies, setStrategies] = useState<StrategyMetadata[]>([])
  const [loading, setLoading] = useState(true)
  const [error, setError] = useState<string | null>(null)

  useEffect(() => {
    const fetchStrategies = async () => {
      try {
        setLoading(true)
        const data = await listStrategies()
        setStrategies(data)
        setError(null)
      } catch (err) {
        setError('Failed to load strategies')
        console.error('Error fetching strategies:', err)
      } finally {
        setLoading(false)
      }
    }

    fetchStrategies()
  }, [])

  if (loading) {
    return <Loading />
  }

  if (error) {
    return <ErrorMessage message={error} />
  }

  if (strategies.length === 0) {
    return (
      <div className="text-center py-8">
        <p className="text-muted-foreground">No strategies available</p>
      </div>
    )
  }

  return (
    <div className="space-y-6">
      <div>
        <h1 className="text-3xl font-bold">Strategies</h1>
        <p className="text-muted-foreground">
          Manage trading strategies including technical analysis, machine learning models, and custom algorithms
        </p>
      </div>

      <div className="grid grid-cols-1 md:grid-cols-2 lg:grid-cols-3 gap-4">
        {strategies.map((strategy) => (
          <StrategyCard
            key={strategy.name}
            strategy={strategy}
          />
        ))}
      </div>
    </div>
  )
}

export default Strategies