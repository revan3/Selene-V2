# Script para limpar locks de banco de dados travados
# Uso: PowerShell -ExecutionPolicy Bypass -File cleanup_locks.ps1

Write-Host "🧹 Limpando locks do Selene..." -ForegroundColor Cyan

# Parar todos os processos
Write-Host "   → Parando processos Selene e Cargo..." -ForegroundColor Yellow
Get-Process | Where-Object {$_.ProcessName -like '*selene*' -or $_.ProcessName -like '*cargo*'} | Stop-Process -Force -ErrorAction SilentlyContinue

Start-Sleep -Seconds 1

# Remover lock files
Write-Host "   → Removendo lock files..." -ForegroundColor Yellow
Remove-Item "F:\Selene_brain_2.0\selene_memories.db\LOCK" -Force -ErrorAction SilentlyContinue

Start-Sleep -Seconds 1

Write-Host "✅ Limpeza concluída! Pronto para rodar:" -ForegroundColor Green
Write-Host "   cargo run --release --bin selene_brain" -ForegroundColor Cyan
