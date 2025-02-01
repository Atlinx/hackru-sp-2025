import { LatLngExpression } from 'leaflet'

import { Category } from './MarkerCategories'

export interface PlaceValues {
  id: number
  position: LatLngExpression
  category: Category
  title: string
  address: string
}
export type PlacesType = PlaceValues[]
export type PlacesClusterType = Record<string, PlaceValues[]>

export const Places: PlacesType = [
  {
    id: 1,
    position: [40.5232513,-74.45816],
    category: Category.CAT1,
    title: 'Busch Student Center',
    address: '604 Bartholomew Rd, Piscataway, NJ 08854',
  },
]
