import * as pkarr from './pkarr.js';
import b4a from 'b4a'

const state = {
  keyPair: {publicKey: null, secretKey: null },
  records: [],
  servers: [
    'http://api.pkarr.nuhvi.com:3000'
  ],
}

init()

function init () {
  generateKeyPairIfMissing() 
  setupTabs()
  setupModal()
  setupRecordsForm()
}

function generateKeyPairIfMissing () {
  if (localStorage.getItem('secretKey')) {
    const secretKey = localStorage.getItem('secretKey')
    state.keyPair = pkarr.keygen(b4a.from(secretKey, 'hex'))
  } else {
    state.keyPair = pkarr.keygen()
    localStorage.setItem('secretKey', b4a.toString(state.keyPair.secretKey, 'hex'))
  }
  console.log(state)

  const publicKey32 = pkarr.encode(state.keyPair.publicKey)
  document.getElementById('public-key').innerHTML = publicKey32
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

function setupModal() {
  const modalContainer = document.getElementById('modal-container');
  const modalContent = document.getElementById('modal-content');
  const modalClose = document.getElementById('modal-close');

  modalClose.addEventListener('click', () => {
    modalContainer.classList.add('hidden');
  });

  const settingsButton = document.getElementById('settings-button');

  settingsButton.addEventListener('click', () => {
    modalContainer.classList.remove('hidden');
  });
}

function setupRecordsForm () {
  const form = document.getElementById('records-form');
  const inputs = form.querySelectorAll('input, textarea, select');
  const size = document.getElementById('size');
  const addRecordButton = document.getElementById('add-record-button');
  const publishButton = document.getElementById('publish-button');

  const stringified = localStorage.getItem('records')
  if (stringified) {
    try {
      const data = JSON.parse(stringified)
      data.forEach((entry) => {
        const inputs = addRecord()
        inputs[0].value = entry[0]
        inputs[1].value = entry[1]
      }) 
      state.records = data
    } catch (error) {}
  } else {
    addRecord()
  }

  publishButton.addEventListener('click', () => {
    publishButton.innerHTML = 'Publishing...'
    publishButton.disabled = true
    pkarr.put(
      state.keyPair,
      state.records,
      state.servers
    ).then((response) => {
      console.log("done", response)
    }).catch((error) => {
      console.log("error", error)
    })
  })

  function onUpdate(x) {
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
    state.records = data
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
