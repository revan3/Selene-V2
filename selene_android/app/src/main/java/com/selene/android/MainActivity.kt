package com.selene.android

import android.Manifest
import android.content.Context
import android.content.pm.PackageManager
import android.net.http.SslError
import android.os.Bundle
import android.view.View
import android.webkit.*
import android.widget.Button
import android.widget.EditText
import android.widget.TextView
import androidx.appcompat.app.AlertDialog
import androidx.appcompat.app.AppCompatActivity
import androidx.core.app.ActivityCompat
import androidx.core.content.ContextCompat

class MainActivity : AppCompatActivity() {

    private lateinit var webView: WebView
    private var permissionCallback: PermissionRequest? = null

    private val PREFS         = "selene_prefs"
    private val KEY_HOST      = "selene_host"
    private val DEFAULT_HOST  = "192.168.1.100:3030"
    private val REQ_PERMS     = 100

    // ── Lifecycle ──────────────────────────────────────────────────────────────

    override fun onCreate(savedInstanceState: Bundle?) {
        super.onCreate(savedInstanceState)

        // Tela de carregamento enquanto WebView inicia
        setContentView(R.layout.activity_main)
        webView = findViewById(R.id.webview)

        configureWebView()

        val host = getHost()
        if (host == DEFAULT_HOST && !hasStoredHost()) {
            showHostDialog(firstTime = true)
        } else {
            loadSelene(host)
        }
    }

    override fun onBackPressed() {
        if (webView.canGoBack()) webView.goBack()
        else super.onBackPressed()
    }

    // ── WebView setup ──────────────────────────────────────────────────────────

    private fun configureWebView() {
        webView.settings.apply {
            javaScriptEnabled          = true
            domStorageEnabled          = true
            databaseEnabled            = true
            allowFileAccess            = false
            mediaPlaybackRequiresUserGesture = false
            mixedContentMode           = WebSettings.MIXED_CONTENT_ALWAYS_ALLOW
            useWideViewPort            = true
            loadWithOverviewMode       = true
            setSupportZoom(false)
            displayZoomControls        = false
            builtInZoomControls        = false
            cacheMode                  = WebSettings.LOAD_DEFAULT
            userAgentString            = "${userAgentString} SeleneAndroid/1.0"
        }

        webView.webViewClient = object : WebViewClient() {
            override fun onReceivedSslError(view: WebView, handler: SslErrorHandler, error: SslError) {
                // Aceita certificados self-signed (para uso com Tailscale / Nginx local)
                handler.proceed()
            }
            override fun shouldOverrideUrlLoading(view: WebView, request: WebResourceRequest): Boolean {
                val url = request.url.toString()
                // Mantém navegação interna — não abre browser externo
                return !url.startsWith("http") && !url.startsWith("ws")
            }
            override fun onPageFinished(view: WebView, url: String) {
                // Injeta o host salvo para que window.location.host funcione corretamente
                val host = getHost()
                view.evaluateJavascript(
                    "if(typeof state!=='undefined' && state.host==='') state.host='$host';",
                    null
                )
            }
        }

        webView.webChromeClient = object : WebChromeClient() {
            // Solicita permissões de câmera e microfone para o WebView
            override fun onPermissionRequest(request: PermissionRequest) {
                val needed = mutableListOf<String>()
                for (res in request.resources) {
                    when (res) {
                        PermissionRequest.RESOURCE_VIDEO_CAPTURE ->
                            if (!hasPerm(Manifest.permission.CAMERA)) needed += Manifest.permission.CAMERA
                        PermissionRequest.RESOURCE_AUDIO_CAPTURE ->
                            if (!hasPerm(Manifest.permission.RECORD_AUDIO)) needed += Manifest.permission.RECORD_AUDIO
                    }
                }
                if (needed.isNotEmpty()) {
                    permissionCallback = request
                    ActivityCompat.requestPermissions(this@MainActivity, needed.toTypedArray(), REQ_PERMS)
                } else {
                    request.grant(request.resources)
                }
            }

            // Exibe título da página na barra de status (opcional)
            override fun onReceivedTitle(view: WebView, title: String) {
                // Silencioso — UI está dentro do WebView
            }
        }
    }

    // ── Carregamento ───────────────────────────────────────────────────────────

    private fun loadSelene(host: String) {
        val url = "http://$host/mobile"
        webView.loadUrl(url)
    }

    // ── Permissões ─────────────────────────────────────────────────────────────

    private fun hasPerm(perm: String) =
        ContextCompat.checkSelfPermission(this, perm) == PackageManager.PERMISSION_GRANTED

    override fun onRequestPermissionsResult(code: Int, perms: Array<out String>, results: IntArray) {
        super.onRequestPermissionsResult(code, perms, results)
        if (code == REQ_PERMS) {
            // Concede os recursos para o WebView mesmo se o usuário negou (o WebView trata o fallback)
            permissionCallback?.grant(permissionCallback?.resources ?: emptyArray())
            permissionCallback = null
        }
    }

    // ── Dialog de configuração de host ────────────────────────────────────────

    private fun getHost(): String =
        getSharedPreferences(PREFS, Context.MODE_PRIVATE).getString(KEY_HOST, DEFAULT_HOST) ?: DEFAULT_HOST

    private fun hasStoredHost(): Boolean =
        getSharedPreferences(PREFS, Context.MODE_PRIVATE).contains(KEY_HOST)

    private fun saveHost(host: String) =
        getSharedPreferences(PREFS, Context.MODE_PRIVATE).edit().putString(KEY_HOST, host).apply()

    private fun showHostDialog(firstTime: Boolean = false) {
        val input = EditText(this)
        input.setText(getHost())
        input.hint = "ex: 192.168.1.100:3030"
        input.setSingleLine()

        val dialog = AlertDialog.Builder(this)
            .setTitle(if (firstTime) "Conectar à Selene" else "Endereço da Selene")
            .setMessage("Informe o IP e porta do servidor Selene na sua rede local.")
            .setView(input)
            .setPositiveButton("Conectar") { _, _ ->
                val host = input.text.toString().trim().trimStart('w', 's', ':', '/')
                if (host.isNotEmpty()) {
                    saveHost(host)
                    loadSelene(host)
                }
            }
            .setCancelable(!firstTime)

        if (!firstTime) {
            dialog.setNeutralButton("Cancelar", null)
        }

        dialog.show()
    }

    // Botão de configuração exposto via menu (opcional — pode ser chamado pelo JS bridge)
    fun onConfigClick(view: View) = showHostDialog()
}
