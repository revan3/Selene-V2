"""
genealogia.py — Banco SQLite que RASTREIA a evolução de código de cada bot.

É o "ponto principal" do Selene-World: mapear as linhagens, registrar QUAIS
refatorações cada uma descobriu, e CATALOGAR as mais úteis — candidatas a portar
pra Selene real. (Ancestralidade rastreada da spec.)

Tabelas:
  bots        — cada indivíduo: pais, geração, latência, código
  descobertas — cada refatoração ADOTADA: transformação + ganho de latência
"""
import os
import sqlite3
import time


class Genealogia:
    def __init__(self, caminho=None):
        self.caminho = caminho or os.path.join(
            os.path.dirname(os.path.abspath(__file__)), "genealogia.db")
        self.con = sqlite3.connect(self.caminho)
        self._criar()

    def _criar(self):
        c = self.con.cursor()
        c.execute("""CREATE TABLE IF NOT EXISTS bots (
            id INTEGER PRIMARY KEY, pai_a INTEGER, pai_b INTEGER,
            geracao INTEGER, nascido REAL, latencia_ms REAL, codigo TEXT)""")
        c.execute("""CREATE TABLE IF NOT EXISTS descobertas (
            id INTEGER PRIMARY KEY AUTOINCREMENT, bot_id INTEGER, geracao INTEGER,
            tarefa TEXT, transformacao TEXT, ganho_pct REAL, ts REAL)""")
        self.con.commit()

    def registrar_bot(self, bot_id, pai_a, pai_b, geracao, latencia_ms, codigo):
        self.con.execute("INSERT OR REPLACE INTO bots VALUES (?,?,?,?,?,?,?)",
                         (bot_id, pai_a, pai_b, geracao, time.time(),
                          latencia_ms, codigo))
        self.con.commit()

    def registrar_descoberta(self, bot_id, ger, tarefa, transformacao, ganho_pct):
        self.con.execute(
            "INSERT INTO descobertas"
            " (bot_id,geracao,tarefa,transformacao,ganho_pct,ts)"
            " VALUES (?,?,?,?,?,?)",
            (bot_id, ger, tarefa, transformacao, ganho_pct, time.time()))
        self.con.commit()

    def catalogo_uteis(self, top=20):
        """Refatorações que mais ajudaram, POR TAREFA — candidatas pra Selene."""
        return self.con.execute("""
            SELECT tarefa, transformacao, COUNT(*) AS vezes,
                   ROUND(AVG(ganho_pct),1) AS ganho_medio,
                   ROUND(MAX(ganho_pct),1) AS ganho_max
            FROM descobertas GROUP BY tarefa, transformacao
            ORDER BY ganho_medio DESC LIMIT ?""", (top,)).fetchall()

    def linhagem(self, bot_id):
        """Ancestralidade (fundador → bot), seguindo o pai_a."""
        out, atual = [], bot_id
        while atual:
            row = self.con.execute(
                "SELECT id,pai_a,pai_b,geracao,latencia_ms FROM bots WHERE id=?",
                (atual,)).fetchone()
            if not row:
                break
            out.append(row)
            atual = row[1]
        return list(reversed(out))

    def total_descobertas(self):
        return self.con.execute("SELECT COUNT(*) FROM descobertas").fetchone()[0]

    def fechar(self):
        self.con.close()
