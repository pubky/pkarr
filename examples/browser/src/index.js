import sodium from 'sodium-universal'
import b4a from 'b4a'
import z32 from 'z32'

const state = {
  server: 'http://api.pkarr.nuhvi.com:3000',
  keyPair: { pk: null, sk: null },
}

const crypto = {
  verify: sodium.crypto_sign_verify_detached,
  sign (msg, sk) {
    const sig = Buffer.alloc(sodium.crypto_sign_BYTES)
    sodium.crypto_sign_detached(sig, msg, sk)
    return sig
  },
  keygen (sk) {
    const pk = Buffer.alloc(sodium.crypto_sign_PUBLICKEYBYTES)
    if (sk == null) {
      sk = sodium.sodium_malloc(sodium.crypto_sign_SECRETKEYBYTES)
      sodium.crypto_sign_keypair(pk, sk)
    } else {
      sodium.crypto_sign_ed25519_sk_to_pk(pk, sk)
    }

    return { pk, sk }
  },
  salt () {
    const s = Buffer.alloc(64)
    sodium.randombytes_buf(s)
    return s
  }
}

init()

function init () {
  state.keyPair = generateKeyPairIfMissing() 
  console.log(state)


  const publicKey32 = z32.encode(b4a.toString(state.keyPair.pk, 'hex'))
  // document.getElementById('public-key').innerHTML = publicKey32
  
  setupTabs()
  setupFooter()
}

function generateKeyPairIfMissing () {
  if (localStorage.getItem('secretKey')) {
    return crypto.keygen(b4a.from(localStorage.getItem('secretKey'), 'hex'))
  }
  const keyPair = crypto.keygen()
  localStorage.setItem('secretKey', b4a.toString(keyPair.sk, 'hex'))

  return keyPair
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

function setupFooter() {
  const serverForm = document.getElementById('server-form');
  serverForm.addEventListener('submit', (e) => {
    e.preventDefault();
    console.log(e)
  })

  document.getElementById('server-input').value = state.server
}
