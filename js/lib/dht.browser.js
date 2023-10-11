/**
 * Mock DHT class for browser environment
 */
export class DHT {
  constructor () {
    throw new Error('not implmented in browser, use Pkarr.relayPut() and Pkarr.relayGet() instead.')
  }
}

export default DHT
