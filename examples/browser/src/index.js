import * as pkarr from './pkarr.js';
import b4a from 'b4a'

const DEFAULT_SERVERS = [
  'http://api.pkarr.nuhvi.com:3000'
]

const DEFAULT_LAST_PUBLISHED = 'Not published yet...'

const store = {
  state: {
    seed: null,
    records: [],
    servers: DEFAULT_SERVERS,
    lastPublished: 'Not published yet'
  },
  load () {
    // SEED
    const seed = localStorage.getItem('seed') 
    if (seed) {
      this.state.seed = seed && b4a.from(seed, 'hex')
    } else {
      this.state.seed = pkarr.randomBytes()
    }
    this.state.keyPair = pkarr.generateKeyPair(this.state.seed)
    // Store to localStorage and trigger render
    this.set('seed', this.state.seed, 'hex')

    // SERVERS
    const servers = tryParseJSON(localStorage.getItem('servers'))
    this.set('servers', servers || DEFAULT_SERVERS)

    // Records
    const records = tryParseJSON(localStorage.getItem('records'))
    this.set('records', records || [])

    // Published date
    const lastPublished = localStorage.getItem('lastPublished') || DEFAULT_LAST_PUBLISHED
    this.set('lastPublished', lastPublished)
  },
  get (key) {
    return this.state[key]
  },
  /** Update both in-memory state and localStorage */
  set (key, value) {
    const encoded = b4a.isBuffer(value) 
      ? b4a.toString(value, 'hex') 
      : typeof value === 'object' 
        ? JSON.stringify(value)
        : value.toString()
    localStorage.setItem(key, encoded)

    if (key === 'seed' && !b4a.equals(this.state.seed, value)) {
      importSeed(value)
    }

    this.state[key] = value
    // Render Dom on change
    render(this.state, key)
    return value
  }
}

const dom = {
  get seed() {
    return document.getElementById('seed');
  },
  get publicKey () {
    return document.getElementById('public-key');
  },
  get servers () {
    return document.getElementById('servers');
  },
  get lastPublished () {
    return document.getElementById('last-published');
  }
}

store.load()

setupModal()
setupTabs()

function render (state, key) {
  switch (key) {
    case 'seed':
      dom.seed.value = b4a.toString(state.seed, 'hex')
      dom.publicKey.innerHTML = 'pk:' + pkarr.encodeID(state.keyPair.publicKey)
      dom.servers.value = state.servers.join('\n')
      break;
    case 'servers':
      dom.servers.value = state.servers.join('\n')
      break;
    case 'records':
      setupRecordsForm(state.records)
      break;
    case 'lastPublished':
      dom.lastPublished.innerHTML = state.lastPublished
      break
  }
}

function setupModal() {
  const settingsButton = document.getElementById('settings-button');
  const modalContainer = document.getElementById('modal-container');
  const modalClose = document.getElementById('modal-close');
  const generateKey = document.getElementById('generate');
  const resetServers = document.getElementById('reset-servers');
  const saveButton = document.getElementById('settings-save-button');

  settingsButton.addEventListener('click', () => {
    modalContainer.classList.remove('hidden');
  });
  generateKey.addEventListener('click', () => {
    dom.seed.value = b4a.toString(pkarr.randomBytes(), 'hex')
  })
  resetServers.addEventListener('click', () => {
    dom.servers.value = DEFAULT_SERVERS.join('\n')
  })
  modalClose.addEventListener('click', () => {
    // Reset seed and servers
    dom.seed.value = b4a.toString(store.get('seed'), 'hex')
    dom.servers.value = store.get('servers').join('\n')
    modalContainer.classList.add('hidden');
  })

  saveButton.addEventListener('click', () => {
    // Save seed and servers
    const seed = b4a.from(dom.seed.value, 'hex')
    const servers = dom.servers.value.split('\n').filter(Boolean)
    store.set('servers', servers)
    store.set('seed', seed)
    modalContainer.classList.add('hidden');
  })
}

function importSeed (seed) {
  console.log("Importing new seed", seed)
  store.state.keyPair = pkarr.generateKeyPair(seed)
  store.set('records', [])
  setupRecordsForm()
  store.set('lastPublished', DEFAULT_LAST_PUBLISHED)
  // TODO: import seed should resolve the thing!
}

function setupTabs() {
  document.getElementById('publish-tab').addEventListener('click', () => {
    setActiveTab('publish');
  });

  document.getElementById('resolve-tab').addEventListener('click', () => {
    setActiveTab('resolve');
  });

  function setActiveTab(tab) {
    const tabButtons = document.getElementsByClassName('tab-button');
    const tabContents = document.getElementsByClassName('tab-content');

    for (let button of tabButtons) {
      button.classList.remove('active');
    }

    for (let content of tabContents) {
      content.classList.add('hidden');
    }

    if (tab === 'publish') {
      document.getElementById('publish-tab').classList.add('active');
      document.getElementById('publish-content').classList.remove('hidden');
    } else if (tab === 'resolve') {
      document.getElementById('resolve-tab').classList.add('active');
      document.getElementById('resolve-content').classList.remove('hidden');
    }
  }
}

function setupRecordsForm () {
  const form = document.getElementById('records-form');
  const inputs = form.querySelectorAll('input, textarea, select');
  const size = document.getElementById('size');
  const addRecordButton = document.getElementById('add-record-button');
  const publishButton = document.getElementById('publish-button');

  // Reset
  form.innerHTML = ""

  const records = store.get('records')
  if (records.length > 0) {
    try {
      records.forEach((entry) => {
        const inputs = addRecord()
        inputs[0].value = entry[0]
        inputs[1].value = entry[1]
      }) 
      store.records = data
    } catch (error) {}
  } else {
    addRecord()
  }

  publishButton.addEventListener('click', () => {
    publishButton.innerHTML = 'Publishing...'
    publishButton.disabled = true

    const ts = Date.now()
    pkarr.put(
      store.get('keyPair'),
      store.get('records'),
      store.get('servers')
    ).then((result) => {
      console.log("PUT time:", Date.now() - ts)
      const lastPublished = new Date(result.seq * 1000).toLocaleString();
      store.set('lastPublished', lastPublished)
    })
    .catch(errors => {
      alert(
        "All servers failed to publish the record:\n" +
        errors.map(error => ` - [${error.server}]: ${error.reason.message}`).join('\n')
      )
    })
    .finally(() => {
      publishButton.innerHTML = 'Publish'
      publishButton.disabled = false
    })
  })

  function onUpdate() {
    const inputs = form.querySelectorAll('input')
    let data = []
    let entry = []
    for (let i=0; i < inputs.length; i++) {
      entry.push(inputs[i].value)
      if (entry.length % 2 === 0) {
        data.push(entry)
        entry = []
      }
    }
    store.records = data
    const stringified = JSON.stringify(data)
    localStorage.setItem('records', stringified)
    size.innerHTML = stringified.length
  }

  inputs.forEach(input => {
    input.addEventListener('input', onUpdate);
    input.addEventListener('change', onUpdate);
  });

  addRecordButton.addEventListener('click', addRecord)

  function addRecord() {
    const row = document.createElement('div');
    const newRow = form.appendChild(row)
    newRow.classList.add('table-row')

    newRow.innerHTML = `
        <input type="text" />
        <input type="text" />
    `
    const newInputs = newRow.querySelectorAll('input');

    newInputs[0].focus()
    newInputs[1].addEventListener('keydown', (e) => {
      if (e.key === 'Enter') {
        addRecord()
      }
    })

    newInputs.forEach(input => {
      input.addEventListener('input', onUpdate);
      input.addEventListener('change', onUpdate);
    })

    return newInputs
  }
}

function tryParseJSON (jsonString) {
  try {
    return JSON.parse(jsonString)
  } catch (error) {
    console.log("error parsing json", {jsonString})
  }
}

// const resolveButton = document.getElementById('resolve')
//
// resolveButton.addEventListener('click', async () => {
//   const value = document.getElementById('query').value;
//   if (!value) return;
//   resolveButton.disabled = true
//
//   const response = await fetch(`${state.server.url}/pkarr/${value}`);
//
//   const json = await response.json();
//
//   resolveButton.disabled = false
//   console.log(json);
// });
//
// Rows([{ type: 'TXT', name: 'Foo', value: 'api.pkarr.nuhvi.com:3000bar', ttl: 3600 }]);
//
// function Rows(rows) {
//   const rowElements = rows.map((row) => {
//     return Row(row);
//   });
//
//   document.getElementById('table').innerHTML = rowElements.join('');
// }
//
// function Row(record) {
//   const rowClass = "border-b border-slate-100 dark:border-slate-700 p-4 text-slate-500 dark:text-slate-400";
//
//   return `
//     <tbody class="bg-white dark:bg-slate-800 flex flex-wrap md:table-row">
//       <tr>
//         <td class="${rowClass} hidden md:block">${record.type}</td>
//         <td class="${rowClass}">${record.name}</td>
//         <td class="${rowClass}">${record.value}</td>
//         <td class="${rowClass} hidden md:block">${record.ttl}</td>
//       </tr>
//     </tbody>
//   `;
// }                
