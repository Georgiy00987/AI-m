"""
ollama_manager.py — обёртка над официальным ollama Python SDK.
Репозиторий SDK: https://github.com/ollama/ollama-python

Установка:
	pip install ollama

Примеры использования:
	# Синхронный
	from ollama_manager import OllamaManager

	with OllamaManager(model="gemma3") as ai:
		print(ai.chat("Привет!"))

		for token in ai.chat_stream("Напиши стихотворение"):
			print(token, end="", flush=True)

		# Диалог с памятью
		ai.ask("Меня зовут Алексей")
		print(ai.ask("Как меня зовут?"))

	# Асинхронный
	from ollama_manager import AsyncOllamaManager
	import asyncio

	async def main():
		async with AsyncOllamaManager(model="gemma3") as ai:
			print(await ai.chat("Привет!"))

	asyncio.run(main())
"""

import os
import subprocess
import time
import atexit
from typing import Iterator, AsyncIterator

import ollama
from ollama import Client, AsyncClient, ChatResponse


# ────────────────────────────────────────────────────────────────────── #
#  Базовый класс                                                         #
# ────────────────────────────────────────────────────────────────────── #

class _BaseManager:
	def __init__(
		self,
		model: str = "gemma3",
		host: str = "http://localhost:11434",
		auto_start: bool = True,
		auto_pull: bool = True,
		stop_on_exit: bool = True,
		system_prompt: str | None = None,
	):
		"""
		Args:
			model:         Модель (gemma3, llama3.2, deepseek-r1, qwen3 и др.)
			host:          Адрес Ollama сервера.
			auto_start:    Запустить ollama serve если не запущена.
			auto_pull:     Скачать модель если не найдена локально.
			stop_on_exit:  Остановить сервер при выходе из контекста
			               (только если мы его сами запустили).
			system_prompt: Системный промпт для всех запросов.
		"""
		self.model = model
		self.host = host
		self.auto_start = auto_start
		self.auto_pull = auto_pull
		self.stop_on_exit = stop_on_exit
		self.system_prompt = system_prompt

		self._process: subprocess.Popen | None = None
		self._started_by_us = False
		self._history: list[dict] = []

	# ── Управление сервером ─────────────────────────────────────────── #

	def _is_running(self) -> bool:
		try:
			Client(host=self.host).list()
			return True
		except Exception:
			return False

	def _get_ollama_path(self) -> str:
		"""Ищет бинарник ollama в PATH и стандартных местах Termux."""
		import shutil
		found = shutil.which("ollama")
		if found:
			return found
		termux_path = "/data/data/com.termux/files/usr/bin/ollama"
		if os.path.isfile(termux_path) and os.access(termux_path, os.X_OK):
			return termux_path
		raise FileNotFoundError(
			"\n❌ Бинарник ollama не найден.\n\n"
			"Установка:\n"
			"  Linux/macOS : curl -fsSL https://ollama.com/install.sh | sh\n"
			"  Windows     : https://ollama.com/download/OllamaSetup.exe\n"
			"  Termux      : pkg install ollama\n"
		)

	def _make_serve_shim(self, ollama_bin: str) -> str:
		"""
		Ollama при запуске модели делает exec("serve", ...) — ищет бинарник
		буквально с именем "serve". Создаём симлинк serve -> ollama во
		временной директории и возвращаем её путь.
		Работает на любой платформе автоматически.
		"""
		import tempfile
		import stat

		tmpdir = tempfile.mkdtemp(prefix="ollama_shim_")
		serve_path = os.path.join(tmpdir, "serve")

		try:
			os.symlink(os.path.abspath(ollama_bin), serve_path)
		except (OSError, NotImplementedError):
			# Симлинки недоступны (редкий случай) — копируем бинарник
			import shutil
			shutil.copy2(ollama_bin, serve_path)
			os.chmod(serve_path, os.stat(serve_path).st_mode | stat.S_IEXEC)

		# Регистрируем очистку при завершении
		atexit.register(lambda: __import__("shutil").rmtree(tmpdir, ignore_errors=True))
		return tmpdir

	def _build_env(self, ollama_bin: str) -> dict:
		"""
		Формирует окружение с правильным PATH для дочернего процесса.
		Добавляет shim-директорию с симлинком serve -> ollama первой в PATH,
		чтобы Ollama могла найти свой runner на любой платформе.
		"""
		env = os.environ.copy()

		shim_dir = self._make_serve_shim(ollama_bin)
		bin_dir = os.path.dirname(os.path.abspath(ollama_bin))

		path_parts = env.get("PATH", "").split(os.pathsep)
		# shim первым, потом реальная директория ollama, потом остальное
		new_parts = [shim_dir, bin_dir] + [
			p for p in path_parts if p and p not in (shim_dir, bin_dir)
		]
		env["PATH"] = os.pathsep.join(new_parts)
		return env

	def _launch_server(self) -> None:
		ollama_bin = self._get_ollama_path()
		env = self._build_env(ollama_bin)

		self._process = subprocess.Popen(
			[ollama_bin, "serve"],
			stdout=subprocess.DEVNULL,
			stderr=subprocess.DEVNULL,
			env=env,
		)
		for _ in range(20):
			time.sleep(1)
			if self._is_running():
				return
		self._process.terminate()
		raise RuntimeError("Ollama не запустилась за 20 секунд.")

	def _ensure_server(self) -> None:
		if self._is_running():
			return
		if not self.auto_start:
			raise RuntimeError(
				"Ollama не запущена. Запустите вручную (`ollama serve`) "
				"или передайте auto_start=True."
			)
		self._launch_server()
		self._started_by_us = True
		atexit.register(self.stop)

	def stop(self) -> None:
		"""Останавливает сервер (только если мы его запустили)."""
		if self._process and self._process.poll() is None:
			self._process.terminate()
			self._process.wait()
			self._process = None

	# ── Управление моделями ─────────────────────────────────────────── #

	def list_models(self) -> list[str]:
		"""Список скачанных моделей."""
		return [m.model for m in Client(host=self.host).list().models]

	def pull(self, model: str | None = None) -> None:
		"""Скачать модель с прогрессом."""
		target = model or self.model
		for chunk in Client(host=self.host).pull(target, stream=True):
			status = chunk.get("status", "")
			completed = chunk.get("completed")
			total = chunk.get("total")
			if total and completed is not None:
				pct = int(completed / total * 100)
				print(f"\r📥 {target}: {pct}%", end="", flush=True)
			elif status:
				print(f"\r📥 {target}: {status}...", end="", flush=True)
		print(f"\r✅ {target} готова!          ")

	def _ensure_model(self) -> None:
		if self.model not in self.list_models():
			if not self.auto_pull:
				raise RuntimeError(
					f"Модель '{self.model}' не найдена. "
					"Скачайте вручную (`ollama pull <model>`) или передайте auto_pull=True."
				)
			self.pull()

	# ── История ─────────────────────────────────────────────────────── #

	def clear_history(self) -> None:
		"""Сбросить историю диалога."""
		self._history.clear()

	def _make_messages(self, prompt: str | None = None) -> list[dict]:
		msgs = []
		if self.system_prompt:
			msgs.append({"role": "system", "content": self.system_prompt})
		msgs.extend(self._history)
		if prompt is not None:
			msgs.append({"role": "user", "content": prompt})
		return msgs


# ────────────────────────────────────────────────────────────────────── #
#  Синхронный менеджер                                                   #
# ────────────────────────────────────────────────────────────────────── #

class OllamaManager(_BaseManager):
	"""Синхронная обёртка над ollama.Client."""

	def __enter__(self) -> "OllamaManager":
		self._ensure_server()
		self._ensure_model()
		self._client = Client(host=self.host)
		return self

	def __exit__(self, *_) -> None:
		if self.stop_on_exit and self._started_by_us:
			self.stop()

	def chat(self, prompt: str, **kwargs) -> str:
		"""Одиночный запрос без сохранения истории."""
		response: ChatResponse = self._client.chat(
			model=self.model,
			messages=self._make_messages(prompt),
			**kwargs,
		)
		return response.message.content

	def chat_stream(self, prompt: str, **kwargs) -> Iterator[str]:
		"""Одиночный запрос, потоковый вывод токенов."""
		stream = self._client.chat(
			model=self.model,
			messages=self._make_messages(prompt),
			stream=True,
			**kwargs,
		)
		for chunk in stream:
			yield chunk["message"]["content"]

	def ask(self, prompt: str, **kwargs) -> str:
		"""Запрос с сохранением истории (многоходовой диалог)."""
		self._history.append({"role": "user", "content": prompt})
		try:
			response: ChatResponse = self._client.chat(
				model=self.model,
				messages=self._make_messages(),
				**kwargs,
			)
			reply = response.message.content
			self._history.append({"role": "assistant", "content": reply})
			return reply
		except Exception:
			self._history.pop()
			raise

	def ask_stream(self, prompt: str, **kwargs) -> Iterator[str]:
		"""Многоходовой запрос, потоковый вывод токенов."""
		self._history.append({"role": "user", "content": prompt})
		full = ""
		try:
			stream = self._client.chat(
				model=self.model,
				messages=self._make_messages(),
				stream=True,
				**kwargs,
			)
			for chunk in stream:
				token = chunk["message"]["content"]
				full += token
				yield token
			self._history.append({"role": "assistant", "content": full})
		except Exception:
			self._history.pop()
			raise

	def generate(self, prompt: str, **kwargs) -> str:
		"""Простой generate без истории и system-промпта."""
		return self._client.generate(model=self.model, prompt=prompt, **kwargs)["response"]

	def embed(self, text: str | list[str], **kwargs):
		"""Эмбеддинги текста."""
		return self._client.embed(model=self.model, input=text, **kwargs)

	def ps(self):
		"""Список запущенных моделей."""
		return self._client.ps()

	def show(self, model: str | None = None):
		"""Информация о модели."""
		return self._client.show(model or self.model)


# ────────────────────────────────────────────────────────────────────── #
#  Асинхронный менеджер                                                  #
# ────────────────────────────────────────────────────────────────────── #

class AsyncOllamaManager(_BaseManager):
	"""Асинхронная обёртка над ollama.AsyncClient."""

	async def __aenter__(self) -> "AsyncOllamaManager":
		self._ensure_server()
		self._ensure_model()
		self._client = AsyncClient(host=self.host)
		return self

	async def __aexit__(self, *_) -> None:
		if self.stop_on_exit and self._started_by_us:
			self.stop()

	async def chat(self, prompt: str, **kwargs) -> str:
		"""Одиночный запрос без сохранения истории."""
		response: ChatResponse = await self._client.chat(
			model=self.model,
			messages=self._make_messages(prompt),
			**kwargs,
		)
		return response.message.content

	async def chat_stream(self, prompt: str, **kwargs) -> AsyncIterator[str]:
		"""Одиночный запрос, асинхронный потоковый вывод."""
		stream = await self._client.chat(
			model=self.model,
			messages=self._make_messages(prompt),
			stream=True,
			**kwargs,
		)
		async for chunk in stream:
			yield chunk["message"]["content"]

	async def ask(self, prompt: str, **kwargs) -> str:
		"""Запрос с сохранением истории."""
		self._history.append({"role": "user", "content": prompt})
		try:
			response: ChatResponse = await self._client.chat(
				model=self.model,
				messages=self._make_messages(),
				**kwargs,
			)
			reply = response.message.content
			self._history.append({"role": "assistant", "content": reply})
			return reply
		except Exception:
			self._history.pop()
			raise

	async def ask_stream(self, prompt: str, **kwargs) -> AsyncIterator[str]:
		"""Многоходовой запрос, асинхронный потоковый вывод."""
		self._history.append({"role": "user", "content": prompt})
		full = ""
		try:
			stream = await self._client.chat(
				model=self.model,
				messages=self._make_messages(),
				stream=True,
				**kwargs,
			)
			async for chunk in stream:
				token = chunk["message"]["content"]
				full += token
				yield token
			self._history.append({"role": "assistant", "content": full})
		except Exception:
			self._history.pop()
			raise

	async def generate(self, prompt: str, **kwargs) -> str:
		response = await self._client.generate(model=self.model, prompt=prompt, **kwargs)
		return response["response"]

	async def embed(self, text: str | list[str], **kwargs):
		return await self._client.embed(model=self.model, input=text, **kwargs)
