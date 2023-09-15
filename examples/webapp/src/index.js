import { render } from 'solid-js/web';
import { createSignal, createEffect, For, onMount } from 'solid-js';
import Pkarr from 'pkarr/relayed.js'
import b4a from 'b4a'

import store from './store.js'

const App = () => {
  const [showSettings, setShowSettings] = createSignal(false);
  const [tab, setTab] = createSignal(0);
  const [target, setTarget] = createSignal(null);
  const [settingsSeed, setSettingsSeed] = createSignal(store.seed)
  const [copySeedText, setCopySeedText] = createSignal('copy seed')
  const [settingsRelays, setSettingsRelays] = createSignal(store.relays)

  createEffect(() => {
    const location = window.location
    const pk = new URL(location.toString()).searchParams.get('pk')
    if (!pk || pk.length !== 52) return
    setTarget(pk)
    setTab(1)
  });

  const toggleSettings = () => {
    setSettingsSeed(store.seed)
    setShowSettings(!showSettings())
  }

  const copySeed = async () => {
    try {
      await navigator.clipboard.writeText(settingsSeed());
      setCopySeedText('Copied...')
    } catch (err) {
      console.error("Error copying text to clipboard:", err);
      setCopySeedText('Errorr copying to clipboard!')
    } finally {
      setTimeout(() => setCopySeedText('copy seed'), 1000)
    }
  }

  const pasteSeed = async () => {
    const seed = await navigator.clipboard.readText();
    if (seed.length !== 64) {
      alert('Seed must be a 64 character hex encoded string')
      return
    }
    setSettingsSeed(seed)
  }

  const handleSaveSettings = () => {
    store.updateSettings(settingsSeed(), settingsRelays())
    setShowSettings(false)
  }

  const handleSettingsRelayChange = (event) => {
    setSettingsRelays(event.target.value.split('\n'))
  }
  const handleResetRelays = () => {
    store.resetRelays()
    setSettingsRelays(store.relays)
  }

  const toggleTab = () => {
    if (tab() === 1) {
      window.location.search = ''
    }
    setTab(tab() === 0 ? 1 : 0)
  }

  return (<>
    <header>
      <div class='row'>
        <a href="https://github.com/nuhvi/pkarr" target="_blank">
          <h1>Pkarr</h1>
        </a>
        <button id="settings-button" title='Configure' onClick={() => toggleSettings()}>
          <svg xmlns="http://www.w3.org/2000/svg" xmlns:xlink="http://www.w3.org/1999/xlink" fill="#000000" version="1.1" id="Capa_1" width="800px" height="800px" viewBox="0 0 45.973 45.973" xml:space="preserve">
            <path d="M43.454,18.443h-2.437c-0.453-1.766-1.16-3.42-2.082-4.933l1.752-1.756c0.473-0.473,0.733-1.104,0.733-1.774    c0-0.669-0.262-1.301-0.733-1.773l-2.92-2.917c-0.947-0.948-2.602-0.947-3.545-0.001l-1.826,1.815    C30.9,6.232,29.296,5.56,27.529,5.128V2.52c0-1.383-1.105-2.52-2.488-2.52h-4.128c-1.383,0-2.471,1.137-2.471,2.52v2.607    c-1.766,0.431-3.38,1.104-4.878,1.977l-1.825-1.815c-0.946-0.948-2.602-0.947-3.551-0.001L5.27,8.205    C4.802,8.672,4.535,9.318,4.535,9.978c0,0.669,0.259,1.299,0.733,1.772l1.752,1.76c-0.921,1.513-1.629,3.167-2.081,4.933H2.501    C1.117,18.443,0,19.555,0,20.935v4.125c0,1.384,1.117,2.471,2.501,2.471h2.438c0.452,1.766,1.159,3.43,2.079,4.943l-1.752,1.763    c-0.474,0.473-0.734,1.106-0.734,1.776s0.261,1.303,0.734,1.776l2.92,2.919c0.474,0.473,1.103,0.733,1.772,0.733    s1.299-0.261,1.773-0.733l1.833-1.816c1.498,0.873,3.112,1.545,4.878,1.978v2.604c0,1.383,1.088,2.498,2.471,2.498h4.128    c1.383,0,2.488-1.115,2.488-2.498v-2.605c1.767-0.432,3.371-1.104,4.869-1.977l1.817,1.812c0.474,0.475,1.104,0.735,1.775,0.735    c0.67,0,1.301-0.261,1.774-0.733l2.92-2.917c0.473-0.472,0.732-1.103,0.734-1.772c0-0.67-0.262-1.299-0.734-1.773l-1.75-1.77    c0.92-1.514,1.627-3.179,2.08-4.943h2.438c1.383,0,2.52-1.087,2.52-2.471v-4.125C45.973,19.555,44.837,18.443,43.454,18.443z     M22.976,30.85c-4.378,0-7.928-3.517-7.928-7.852c0-4.338,3.55-7.85,7.928-7.85c4.379,0,7.931,3.512,7.931,7.85    C30.906,27.334,27.355,30.85,22.976,30.85z" />
          </svg>
        </button>
      </div>
      <p>Resolvable sovereign keys, now!</p>
    </header>

    <main>
      <div class="tabs">
        <button id="publish-tab" class={`tab-button ${tab() === 0 && 'active'}`} onClick={toggleTab}>Publish</button>
        <button id="resolve-tab" class={`tab-button ${tab() === 1 && 'active'}`} onClick={toggleTab}>Resolve</button>
      </div>

      <article id="publish-content" class={"tab-content" + ` ${tab() === 1 && 'hidden'}`}>
        <h3 id='public-key'>{store.pk()}</h3>
        <Records />
        <ul class="fine-text">
          <li>Add some records to your public-key, and publish it to the DHT when you are ready.</li>
          <li>Copy your key `pk:...` and share with your friends so they can fecth and watch your records.</li>
          <li>Paste any of your friend's keys in the Resolve tab to get their latest records.</li>
          <li>You need to republish your records every hour or so, otherwise they will be dropped from the DHT.</li>
          <li>Anyone can republish your records, and you can republish your friends records too.</li>
          <li>Open settings to add or remove relays, you can also <a href="https://github.com/Nuhvi/pkarr/tree/main/js#relays" target="_blank">run your own relay</a>.</li>
          <li>Open settings to export, import, or regenerate new seed.</li>
          <li>It is safe to edit your records from multiple devices, but some changes might be lost.</li>
        </ul>
      </article>

      <article id="resolve-content" class={"tab-content" + ` ${tab() === 0 && 'hidden'}`}>
        <h3>Resolve</h3>
        <input id="resolve-input" autofocus placeholder="paste a public-key to lookup and resolve" onInput={(e) => setTarget(e.target.value)} value={target()}></input>
        <Records resolver target={target} />
      </article>
    </main >

    <div id='modal-container' class={!showSettings() && 'hidden'}>
      <div id='modal-content'>
        <div class='row'>
          <h2>Configure</h2>
          <button id='modal-close' onClick={() => toggleSettings()}>&times;</button>
        </div>
        <div class="modal-body" id="settings">
          <div class="row">
            <label>Seed</label>
            <input id="seed" value={settingsSeed().slice(0, 8) + "*".repeat(56)}></input>
            <div id="seed-buttons">
              <button id="paste-seed" onClick={pasteSeed}>paste seed</button>
              <button id="copy-seed" onClick={copySeed}>{copySeedText()}</button>
              <button id="generate-seed" onClick={() => setSettingsSeed(b4a.toString(Pkarr.generateSeed(), 'hex'))}>generate</button>
            </div>
          </div>
          <div class="row">
            <label>Relays</label>
            <textarea id="relays" rows="5" onChange={handleSettingsRelayChange}>{settingsRelays().join('\n')}</textarea>
            <button id='reset-relays' onClick={handleResetRelays}>reset</button>
          </div>
          <button class="primary" id="settings-save-button" onClick={handleSaveSettings}>Save</button>
        </div>
      </div>
    </div>

    <footer>
      <p>
        This is a proof of concept for demonstration purposes only.
      </p>
      <p>
        Read the motivation, architecture and how it works, <a href="https://github.com/nuhvi/pkarr" target="_blank">README</a>
      </p>
      <p>
        Open an <a href="https://github.com/nuhvi/pkarr/issues" target="_blank">Issue</a> offering a free relay, your expertise, or help shape the spec.
      </p>
    </footer>

  </>
  );
};

function Records({ resolver, target }) {
  let form;

  function addRecord(e) {
    e?.preventDefault()
    store.addRecord()
    const inputs = form.querySelectorAll('input')
    const input = inputs[inputs.length - 2]
    input.focus()
  }

  function onKeyDown(x) {
    if (x.code === 'Enter') {
      addRecord()
    }
  }

  function handlePublish(e) {
    e?.preventDefault();
    store.publish()
  }

  function handleResolve(e) {
    e?.preventDefault();
    store.resolve(target())
    window.history.replaceState(null, '', `?pk=${target().replace('pk:', '')}`)
  }

  onMount(() => {
    const _target = target?.()
    if (!_target) return
    store.resolve(_target)
  })

  let typingTimer;


  function handleInput() {
    clearTimeout(typingTimer);
    typingTimer = setTimeout(
      () => {
        const records = [...form.querySelectorAll('input')]
          .reduce((acc, input, i) => {
            if (i % 2 === 0) {
              acc.push([input.value])
            } else {
              acc[acc.length - 1].push(input.value)
            }
            return acc
          }, [])
        store.updateRecords(records)
      },
      1000
    )
  }

  return <div class="table">
    <div class="table-header">
      <div class="table-header-cell">Name</div>
      <div class="table-header-cell">Value</div>
    </div>
    <div class='records'>
      <form ref={form} id="records-form" onKeyDown={onKeyDown} >
        <For each={resolver ? store.resolved : store.records}>
          {(row, rowIndex) => {
            return <div class="table-row">
              <input
                type="text"
                disabled={resolver}
                placeholder={!resolver ? 'name' : 'No records yet...'}
                value={row[0] || ""}
                onInput={(e) => handleInput(e, rowIndex(), 0)}
                autofocus
              />
              <input
                type="text"
                disabled={resolver}
                placeholder={!resolver ? 'value' : ''}
                value={row[1] || ""}
                onInput={(e) => handleInput(e, rowIndex(), 1)}
              />
            </div>
          }}
        </For>
      </form>
      <div class="details">
        <div><b>last update: </b><span id="last-published">{resolver ? store.resolvedLastPublished : store.lastPublished}</span></div>
        <div><b>compressed size: </b> <span id='size'>{resolver ? store.resolvedSize : store.recordsSize}</span>/1000 bytes</div>
      </div>
      <div id="add-record-button-container">
        {!resolver && <button id="add-record-button" onClick={addRecord}><span>+</span> Add Record</button>}
      </div>
      <div class="buttons-record-container">
        <button id="publish-button" class="primary"
          disabled={resolver ? store.resolving : store.publishing}
          onClick={resolver ? handleResolve : handlePublish}
        >
          {
            (resolver
              ? store.resolving ? 'Resolving... this may take few seconds' : 'Resolve'
              : store.publishing ? 'Publishing... this may take few seconds' : 'Publish'
            )
            + (store.temporaryMessage || "")
          }
        </button>
      </div>
    </div>
  </div>
}

const app = document.getElementById('app');

if (app) {
  render(() => <App />, app);
}
