@echo off
title Selene Brain 2.0 — Acesso Remoto
color 0B

echo.
echo  ============================================================
echo   SELENE BRAIN 2.0 — MODO REMOTO (acesso pelo celular)
echo  ============================================================
echo.

REM ── Verifica se o binario release existe ─────────────────────
if not exist "target\release\selene_brain.exe" (
    echo  [!] Binario nao encontrado. Compilando em release...
    cargo build --release
    if errorlevel 1 ( echo  [ERRO] Compilacao falhou. & pause & exit /b 1 )
)

REM ── Inicia a Selene em background ────────────────────────────
echo  [1/2] Iniciando Selene Brain na porta 3030...
start "Selene Brain" /MIN "target\release\selene_brain.exe"
timeout /t 4 /nobreak >nul

echo  [2/2] Iniciando tunel HTTPS...
echo.

REM ─── OPCAO A: cloudflared (SEM CONTA, SEM LOGIN) ─────────────
where cloudflared >nul 2>&1
if %errorlevel%==0 (
    echo  Usando Cloudflare Tunnel (sem conta necessaria)...
    echo.
    echo  Aguarde a URL aparecer na linha "trycloudflare.com"...
    echo  Ex:  https://exemplo-nome-aleatorio.trycloudflare.com
    echo.
    echo  No celular acesse: [URL]/mobile
    echo  Para instalar como app: "Adicionar a tela inicial" no browser
    echo.
    cloudflared tunnel --url http://localhost:3030
    goto :fim
)

REM ─── OPCAO B: ngrok (REQUER CONTA GRATUITA + AUTHTOKEN) ──────
where ngrok >nul 2>&1
if %errorlevel%==0 (
    REM Verifica se authtoken ja foi configurado
    ngrok config check >nul 2>&1
    if %errorlevel%==0 (
        echo  Usando ngrok...
        echo  Aguarde a URL https:// aparecer na janela do ngrok.
        echo  No celular acesse: [URL]/mobile
        echo.
        ngrok http 3030
        goto :fim
    ) else (
        echo  [!] ngrok encontrado mas sem authtoken configurado.
        echo.
        echo  Para usar o ngrok:
        echo    1. Crie conta gratis em https://dashboard.ngrok.com/signup
        echo    2. Copie seu token em https://dashboard.ngrok.com/get-started/your-authtoken
        echo    3. Execute: ngrok config add-authtoken SEU_TOKEN
        echo    4. Rode este script novamente.
        echo.
    )
)

REM ─── NENHUM TUNEL ENCONTRADO ─────────────────────────────────
echo  Nenhum programa de tunel encontrado. Escolha uma opcao:
echo.
echo  OPCAO RECOMENDADA — Cloudflare Tunnel (gratuito, sem conta):
echo    1. Baixe em: https://github.com/cloudflare/cloudflared/releases/latest
echo       Arquivo: cloudflared-windows-amd64.exe
echo    2. Renomeie para: cloudflared.exe
echo    3. Coloque nesta pasta: %~dp0
echo    4. Execute este script novamente.
echo.
echo  OPCAO ALTERNATIVA — Acesso so na rede local (sem instalar nada):
for /f "tokens=2 delims=:" %%a in ('ipconfig ^| findstr /i "IPv4"') do (
    set LOCAL_IP=%%a
    goto :show_ip
)
:show_ip
set LOCAL_IP=%LOCAL_IP: =%
echo    No celular (mesma rede Wi-Fi): http://%LOCAL_IP%:3030/mobile
echo.

:fim
pause
