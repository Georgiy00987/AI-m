"""
tg_auto_post.py — автопостинг в Telegram-канал через aiogram.

Установка:
	pip install ollama requests aiogram python-dotenv

.env файл:
	BOT_TOKEN_CHANNEL=123456:ABC...
	GITHUB_REPO_CHANNEL=username/repository
	GITHUB_TOKEN=ghp_...
	CHANNEL_URL=https://t.me/mychannel

История постов хранится в файле sent_posts.log прямо в GitHub-репозитории.

Запуск:
	python tg_auto_post.py
"""

import asyncio
import base64
import json
import os
import requests
from dotenv import load_dotenv
from aiogram import Bot
from aiogram.enums import ParseMode
from aiogram.client.default import DefaultBotProperties

from ollama_manager import OllamaManager


load_dotenv()

BOT_TOKEN_CHANNEL = os.getenv("BOT_TOKEN_CHANNEL")   # токен бота
GITHUB_REPO_CHANNEL      = os.getenv("GITHUB_REPO_CHANNEL", "")    # формат: username/repo
GITHUB_TOKEN     = os.getenv("GITHUB_TOKEN", "")   # ghp_...
CHANNEL_URL      = os.getenv("CHANNEL_URL", "")    # https://t.me/mychannel

# ID канала извлекаем из CHANNEL_URL: https://t.me/mychannel -> @mychannel
CHANNEL = "@" + CHANNEL_URL.rstrip("/").split("/")[-1] if CHANNEL_URL else ""

MODEL         = "qwen2.5:0.5b"
HISTORY_LIMIT = 50
HISTORY_PATH  = "sent_posts.log"   # путь к файлу внутри GitHub-репозитория


# ── Валидация env ─────────────────────────────────────────────────────── #

def validate_env() -> None:
	missing = [k for k, v in {
		"BOT_TOKEN_CHANNEL": BOT_TOKEN_CHANNEL,
		"GITHUB_REPO_CHANNEL":      GITHUB_REPO_CHANNEL,
		"GITHUB_TOKEN":     GITHUB_TOKEN,
		"CHANNEL_URL":      CHANNEL_URL,
	}.items() if not v]
	if missing:
		raise RuntimeError(f"❌ Не заданы переменные окружения: {', '.join(missing)}")


# ── GitHub API — базовый хелпер ───────────────────────────────────────── #

def _gh_headers() -> dict:
	return {
		"Accept":        "application/vnd.github+json",
		"Authorization": f"Bearer {GITHUB_TOKEN}",
	}


# ── История постов в GitHub ───────────────────────────────────────────── #

def _get_file_info(path: str) -> dict | None:
	"""Возвращает содержимое файла из репозитория (content + sha)."""
	resp = requests.get(
		f"https://api.github.com/repos/{GITHUB_REPO_CHANNEL}/contents/{path}",
		headers=_gh_headers(),
		timeout=10,
	)
	if resp.status_code == 404:
		return None
	resp.raise_for_status()
	return resp.json()


def load_history() -> tuple[list[str], str | None]:
	"""
	Загружает историю постов из GitHub.
	Возвращает (список строк, sha файла для обновления).
	sha нужен для PUT-запроса при обновлении файла.
	"""
	print("   Читаю историю из GitHub...")
	info = _get_file_info(HISTORY_PATH)
	if not info:
		print("   Файл истории не найден, начинаем с нуля")
		return [], None

	content = base64.b64decode(info["content"]).decode("utf-8")
	lines = [l.strip() for l in content.splitlines() if l.strip()]
	sha = info["sha"]
	return lines[-HISTORY_LIMIT:], sha


def save_to_history(text: str, sha: str | None) -> None:
	"""Сохраняет новую запись в файл истории на GitHub (create или update)."""
	print("   Сохраняю историю на GitHub...")

	# Получаем актуальное содержимое (sha может измениться)
	info = _get_file_info(HISTORY_PATH)
	current_lines = []
	current_sha = sha
	if info:
		current_lines = base64.b64decode(info["content"]).decode("utf-8").splitlines()
		current_sha = info["sha"]

	current_lines.append(text[:200].replace("\n", " "))
	new_content = "\n".join(current_lines) + "\n"
	encoded = base64.b64encode(new_content.encode("utf-8")).decode("utf-8")

	payload: dict = {
		"message": "chore: update post history",
		"content": encoded,
	}
	if current_sha:
		payload["sha"] = current_sha

	resp = requests.put(
		f"https://api.github.com/repos/{GITHUB_REPO_CHANNEL}/contents/{HISTORY_PATH}",
		headers=_gh_headers(),
		json=payload,
		timeout=15,
	)
	resp.raise_for_status()
	print("   ✅ История сохранена в GitHub")


# ── GitHub — информация о репозитории ────────────────────────────────── #

def fetch_github_repo_info() -> dict | None:
	if not GITHUB_REPO_CHANNEL or "/" not in GITHUB_REPO_CHANNEL:
		return None
	try:
		resp = requests.get(
			f"https://api.github.com/repos/{GITHUB_REPO_CHANNEL}",
			headers=_gh_headers(),
			timeout=10,
		)
		if resp.status_code != 200:
			print(f"⚠️  GitHub repo info: {resp.status_code}")
			return None
		r = resp.json()
		return {
			"full_name":   r["full_name"],
			"description": r.get("description") or "Без описания",
			"url":         r["html_url"],
			"language":    r.get("language") or "Не указан",
			"stars":       r["stargazers_count"],
			"forks":       r["forks_count"],
			"topics":      r.get("topics", []),
			"updated":     r["updated_at"][:10],
		}
	except Exception as e:
		print(f"⚠️  Ошибка получения репо: {e}")
		return None


def fetch_recent_commits(limit: int = 5) -> list[str]:
	try:
		resp = requests.get(
			f"https://api.github.com/repos/{GITHUB_REPO_CHANNEL}/commits",
			headers=_gh_headers(),
			params={"per_page": limit},
			timeout=10,
		)
		if resp.status_code != 200:
			return []
		return [c["commit"]["message"].split("\n")[0] for c in resp.json()]
	except Exception:
		return []


# ── Генерация поста ───────────────────────────────────────────────────── #

SYSTEM_PROMPT = """Ты — автор Telegram-канала о Python и GitHub.
Пишешь посты на русском языке, интересно и по делу.

Формат поста (Telegram HTML):
- Первая строка: <b>🔥 Заголовок</b>
- Пустая строка
- Основной текст, 2-3 абзаца. Важные слова в <b>тегах</b>
- Примеры кода в <code>тегах</code>
- Пустая строка
- Хэштеги: #Python #GitHub и по теме
- Последняя строка: 👉 {channel_url}

Длина: 150-250 слов. Без вступлений — сразу к делу."""


def build_prompt(past_posts: list[str], repo_info: dict | None, commits: list[str]) -> str:
	past_str = (
		"\n".join(f"- {p}" for p in past_posts)
		if past_posts else "постов ещё не было"
	)

	repo_str = ""
	if repo_info:
		repo_str = f"""
Информация о GitHub-репозитории:
- Название: {repo_info['full_name']}
- Описание: {repo_info['description']}
- Язык: {repo_info['language']}
- Звёзды: {repo_info['stars']} | Форки: {repo_info['forks']}
- Теги: {', '.join(repo_info['topics']) or 'нет'}
- Ссылка: {repo_info['url']}
- Обновлён: {repo_info['updated']}"""

	commits_str = ""
	if commits:
		commits_str = "\nПоследние коммиты:\n" + "\n".join(f"- {c}" for c in commits)

	return f"""Темы уже опубликованных постов (не повторяй):
{past_str}
{repo_str}
{commits_str}

Задача:
1. Выбери новую уникальную тему:
   - Что нового в репозитории (по коммитам)
   - Как использовать этот проект
   - Полезная Python-библиотека или инструмент
   - Лайфхак, паттерн, фича Python
   - Совет по Git/GitHub
2. Напиши готовый пост. Не объясняй выбор — сразу пиши пост."""


def generate_post(past_posts: list[str], repo_info: dict | None, commits: list[str]) -> str:
	system = SYSTEM_PROMPT.replace("{channel_url}", CHANNEL_URL)
	prompt = build_prompt(past_posts, repo_info, commits)
	with OllamaManager(model=MODEL, system_prompt=system) as ai:
		return ai.chat(prompt)


# ── Отправка в Telegram ───────────────────────────────────────────────── #

async def send_post(text: str) -> None:
	bot = Bot(
		token=BOT_TOKEN_CHANNEL,
		default=DefaultBotProperties(parse_mode=ParseMode.HTML),
	)
	async with bot:
		try:
			await bot.send_message(chat_id=CHANNEL, text=text)
			print("✅ Пост отправлен!")
		except Exception as e:
			print(f"⚠️  HTML не прошёл ({e}), отправляю plain text")
			await bot.send_message(chat_id=CHANNEL, text=text, parse_mode=None)
			print("✅ Пост отправлен (plain text)!")


# ── Главная функция ───────────────────────────────────────────────────── #

async def main() -> None:
	validate_env()

	print("📖 Загружаю историю постов...")
	past_posts, history_sha = load_history()
	print(f"   Найдено: {len(past_posts)} постов")

	print(f"📦 Получаю данные с GitHub: {GITHUB_REPO_CHANNEL}")
	repo_info = fetch_github_repo_info()
	commits = fetch_recent_commits()
	if repo_info:
		print(f"   ✅ {repo_info['full_name']} — ⭐{repo_info['stars']}")
	print(f"   Последних коммитов: {len(commits)}")

	print("🤖 Генерирую пост...")
	post = generate_post(past_posts, repo_info, commits)

	print("\n── Сгенерированный пост ──────────────────────────────")
	print(post)
	print("──────────────────────────────────────────────────────\n")

	await send_post(post)
	save_to_history(post, history_sha)


if __name__ == "__main__":
	asyncio.run(main())
