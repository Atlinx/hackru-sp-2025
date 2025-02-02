import Head from 'next/head'

import Map from '#components/Map'

const MapPage = () => (
  <div>
    <Head>
      <title>doodlebob</title>
      <meta
        property="og:title"
        content="doodlebob"
        key="title"
      />
      <meta
        name="description"
        content="doodlebob."
      />
    </Head>
    <Map />
  </div>
)

export default MapPage
