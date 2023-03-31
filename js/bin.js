#!/usr/bin/env node
import pm2 from 'pm2'
import z32 from 'z32'
import pkarr from './index.js'

const serverName = 'pkarr-server'

const runServer = () => {
  pm2.connect((err) => {
    if (err) {
      console.error('Error connecting to PM2:', err)
      process.exit(2)
    }
    pm2.start({
      script: './start.js',
      name: serverName
    }, (err) => {
      pm2.disconnect()
      if (err) {
        console.error('Error starting Pkarr server with PM2:', err)
        process.exit(2)
      }
      console.log('Pkarr server started using PM2')
    })
  })
}

const stopServer = () => {
  pm2.connect((err) => {
    if (err) {
      console.error('Error connecting to PM2:', err)
      process.exit(2)
    }
    pm2.stop(serverName, (err) => {
      pm2.disconnect()
      if (err) {
        console.error('Error stopping Pkarr server with PM2:', err)
        process.exit(2)
      }
      console.log('Pkarr server stopped using PM2')
    })
  })
}

const checkStatus = () => {
  pm2.connect((err) => {
    if (err) {
      console.error('Error connecting to PM2:', err)
      process.exit(2)
    }
    pm2.describe(serverName, (err, processDescription) => {
      pm2.disconnect()
      if (err) {
        console.error('Error getting Pkarr server status with PM2:', err)
        process.exit(2)
      }

      if (processDescription[0] && processDescription[0].pm2_env.status === 'online') {
        console.log('Pkarr server is running')
      } else {
        console.log('Pkarr server is not running')
      }
    })
  })
}

const resolveKey = async (key, server = 'http://0.0.0.0:7527') => {
  const keyBytes = z32.decode(key.replace('pk:', ''))
  try {
    const result = await pkarr.get(keyBytes, [server])
    if (result.ok) {
      console.log('Resolved key', key, 'to:\n', JSON.stringify({
        last_updated: new Date(result.seq * 1000).toLocaleString(),
        records: result.records
      }, null, 2))
    } else {
      console.log('Erorr resolving key', key, result.errors[0])
    }
  } catch (error) {
    console.log('Erorr resolving key', key, error)
  }
}

const showHelp = () => {
  console.log(`
Usage: pkarr [command] [options]

Commands:
  run                    Run Pkarr server using PM2 (has to listen on a publicly addressable IP)
  resolve <key> [server] Make a request to the server (default to 0.0.0.0:7527) and log the response
  stop                   Stop the server server using PM2
  status                 Check the status of the server process
  help                   Show this help message

Options:
  <key>              The key to be resolved by the 'resolve' command

Examples:
  pkarr run
  pkarr resolve example-key
  pkarr stop
  pkarr status
  pkarr help
`)
}

const command = process.argv[2]

switch (command) {
  case 'run':
    runServer()
    break
  case 'resolve':
    const [, , , key, server] = process.argv
    if (!key) {
      console.error('Please provide a key to resolve')
      process.exit(1)
    }
    resolveKey(key, server)
    break
  case 'stop':
    stopServer()
    break
  case 'status':
    checkStatus()
    break
  case '-h':
  case 'help':
    showHelp()
    break
  default:
    console.error('Invalid command')
    process.exit(1)
}
