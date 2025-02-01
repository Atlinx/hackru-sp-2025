import { Leaf } from 'lucide-react'
import Head from 'next/head'
import Link from 'next/link'

import NavMenu from '#components/common/NavMenu'
import { AppConfig } from '#lib/AppConfig'

const Home = () => (
  <div className="container mx-auto max-w-2xl p-3 max-md:max-w-none">
    <Head>
      <title>Doodlebob</title>
      <meta
        property="og:title"
        content="Doodlebob"
        key="title"
      />
      <meta
        name="description"
        content="adaptive bus sim"
      />
    </Head>
    <header className="items-top mt-10 gap-4 md:flex">
    </header>
    <section>
      <p className="my-3">
      </p>
    </section>
    <section className="grid grid-cols-1 md:grid-cols-2">
      <div>
        <NavMenu />
      </div>
    </section>
  </div>
)

export default Home
