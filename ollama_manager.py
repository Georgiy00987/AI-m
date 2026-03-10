"""
ollama_manager.py — обёртка над официальным ollama Python SDK.
Репозиторий SDK: https://github.com/ollama/ollama-python

Установка:
	pip install ollama

Два режима работы:
	1. Локальный  — ollama запущена на том же сервере
	2. Облачный   — модели работают на ollama.com (не нужен диск и RAM)

.env для облачного режима:
	OLLAMA_API_KEY=ваш_ключ  # https://ollama.com/settings/keys
	OLLAMA_CLOUD=true

Примеры использования:
	from ollama_manager import OllamaManager

	# Локально
	with OllamaManager(model="qwen2.5:0.5b") as ai:
		print(ai.chat("Привет!"))

	# Облачно (без локальной установки ollama)
	with OllamaManager(model="gemma3", cloud=True) as ai:
		print(ai.chat("Привет!"))
"""

import os
import subprocess
import time
import atexit
from typing import Iterator, AsyncIterator

from ollama import Client, AsyncClient, ChatResponse


# ────────────────────────────────────────────────────────────────────── #
#  Базовый класс                                                         #
# ────────────────────────────────────────────────────────────────────── #

class _BaseManager:
	CLOUD_HOST = "https://ollama.com"

	def __init__(
		self,
		model: str = "gemma3",
		host: str = "http://localhost:11434",
		auto_start: bool = True,
		auto_pull: bool = True,
		stop_on_exit: bool = True,
		system_prompt: str | None = None,
		cloud: bool = False,
		api_key: str | None = None,
	):
		"""
		Args:
			model:         Модель (gemma3, llama3.2, qwen2.5:0.5b и др.)
			host:          Адрес локального Ollama сервера.
			auto_start:    Запустить ollama serve если не запущена (только локально).
			auto_pull:     Скачать модель если не найдена (только локально).
			stop_on_exit:  Остановить сервер при выходе из контекста.
			system_prompt: Системный промпт для всех запросов.
			cloud:         Использовать облачный API ollama.com.
			api_key:       API-ключ для облака (или из env OLLAMA_API_KEY).
		"""
		self.model        = model
		self.host         = host
		self.auto_start   = auto_start
		self.auto_pull    = auto_pull
		self.stop_on_exit = stop_on_exit
		self.system_prompt = system_prompt

		# Облачный режим: из аргумента или из env OLLAMA_CLOUD=true
		self.cloud = cloud or os.getenv("OLLAMA_CLOUD", "").lower() == "true"
		self.api_key = api_key or os.getenv("OLLAMA_API_KEY", "")

		self._process: subprocess.Popen | None = None
		self._started_by_us = False
		self._history: list[dict] = []

	# ── Создание клиента ────────────────────────────────────────────── #

	def _make_client(self) -> Client:
		if self.cloud:
			if not self.api_key:
				raise RuntimeError(
					"❌ Облачный режим требует OLLAMA_API_KEY.\n"
					"Получите ключ на https://ollama.com/settings/keys\n"
					"и добавьте в .env: OLLAMA_API_KEY=ваш_ключ"
				)
			return Client(
				host=self.CLOUD_HOST,
				headers={"Authorization": f"Bearer {self.api_key}"},
			)
		return Client(host=self.host)

	def _make_async_client(self) -> AsyncClient:
		if self.cloud:
			if not self.api_key:
				raise RuntimeError(
					"❌ Облачный режим требует OLLAMA_API_KEY.\n"
					"Получите ключ на https://ollama.com/settings/keys"
				)
			return AsyncClient(
				host=self.CLOUD_HOST,
				headers={"Authorization": f"Bearer {self.api_key}"},
			)
		return AsyncClient(host=self.host)

	# ── Управление локальным сервером ───────────────────────────────── #

	def _is_running(self) -> bool:
		try:
			Client(host=self.host).list()
			return True
		except Exception:
			return False

	def _get_ollama_path(self) -> str:
		import shutil
		found = shutil.which("ollama")
		if found:
			return found
		# Termux
		termux_path = "/data/data/com.termux/files/usr/bin/ollama"
		if os.path.isfile(termux_path) and os.access(termux_path, os.X_OK):
			return termux_path
		raise FileNotFoundError(
			"\n❌ Бинарник ollama не найден.\n\n"
			"Варианты:\n"
			"  Linux/macOS : curl -fsSL https://ollama.com/install.sh | sh\n"
			"  Windows     : https://ollama.com/download/OllamaSetup.exe\n"
			"  Termux      : pkg install ollama\n"
			"  Без установки: передайте cloud=True для работы через ollama.com\n"
		)

	def _make_serve_shim(self, ollama_bin: str) -> str:
		import tempfile, stat
		tmpdir = tempfile.mkdtemp(prefix="ollama_shim_")
		serve_path = os.path.join(tmpdir, "serve")
		try:
			os.symlink(os.path.abspath(ollama_bin), serve_path)
		except (OSError, NotImplementedError):
			import shutil
			shutil.copy2(ollama_bin, serve_path)
			os.chmod(serve_path, os.stat(serve_path).st_mode | stat.S_IEXEC)
		atexit.register(lambda: __import__("shutil").rmtree(tmpdir, ignore_errors=True))
		return tmpdir

	def _build_env(self, ollama_bin: str) -> dict:
		env = os.environ.copy()
		shim_dir = self._make_serve_shim(ollama_bin)
		bin_dir = os.path.dirname(os.path.abspath(ollama_bin))
		path_parts = env.get("PATH", "").split(os.pathsep)
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
		if self.cloud:
			return  # облако — сервер не нужен
		if self._is_running():
			return
		if not self.auto_start:
			raise RuntimeError(
				"Ollama не запущена. Запустите вручную или передайте auto_start=True.\n"
				"Для работы без установки используйте cloud=True."
			)
		self._launch_server()
		self._started_by_us = True
		atexit.register(self.stop)

	def stop(self) -> None:
		if self._process and self._process.poll() is None:
			self._process.terminate()
			self._process.wait()
			self._process = None

	# ── Управление моделями ─────────────────────────────────────────── #

	def list_models(self) -> list[str]:
		return [m.model for m in self._make_client().list().models]

	def pull(self, model: str | None = None) -> None:
		if self.cloud:
			print("ℹ️  Облачный режим — модели не скачиваются локально")
			return
		target = model or self.model
		for chunk in self._make_client().pull(target, stream=True):
			status    = chunk.get("status", "")
			completed = chunk.get("completed")
			total     = chunk.get("total")
			if total and completed is not None:
				print(f"\r📥 {target}: {int(completed / total * 100)}%", end="", flush=True)
			elif status:
				print(f"\r📥 {target}: {status}...", end="", flush=True)
		print(f"\r✅ {target} готова!          ")

	def _ensure_model(self) -> None:
		if self.cloud:
			return  # облако — модель не нужно скачивать
		if self.model not in self.list_models():
			if not self.auto_pull:
				raise RuntimeError(
					f"Модель '{self.model}' не найдена. "
					"Скачайте вручную или передайте auto_pull=True."
				)
			self.pull()

	# ── История диалога ─────────────────────────────────────────────── #

	def clear_history(self) -> None:
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
	"""Синхронная обёртка над ollama.Client (локальный + облачный режим)."""

	def __enter__(self) -> "OllamaManager":
		self._ensure_server()
		self._ensure_model()
		self._client = self._make_client()
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
		return self._client.generate(model=self.model, prompt=prompt, **kwargs)["response"]

	def embed(self, text: str | list[str], **kwargs):
		return self._client.embed(model=self.model, input=text, **kwargs)

	def ps(self):
		return self._client.ps()

	def show(self, model: str | None = None):
		return self._client.show(model or self.model)


# ────────────────────────────────────────────────────────────────────── #
#  Асинхронный менеджер                                                  #
# ────────────────────────────────────────────────────────────────────── #

class AsyncOllamaManager(_BaseManager):
	"""Асинхронная обёртка над ollama.AsyncClient (локальный + облачный режим)."""

	async def __aenter__(self) -> "AsyncOllamaManager":
		self._ensure_server()
		self._ensure_model()
		self._client = self._make_async_client()
		return self

	async def __aexit__(self, *_) -> None:
		if self.stop_on_exit and self._started_by_us:
			self.stop()

	async def chat(self, prompt: str, **kwargs) -> str:
		response: ChatResponse = await self._client.chat(
			model=self.model,
			messages=self._make_messages(prompt),
			**kwargs,
		)
		return response.message.content

	async def chat_stream(self, prompt: str, **kwargs) -> AsyncIterator[str]:
		stream = await self._client.chat(
			model=self.model,
			messages=self._make_messages(prompt),
			stream=True,
			**kwargs,
		)
		async for chunk in stream:
			yield chunk["message"]["content"]

	async def ask(self, prompt: str, **kwargs) -> str:
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
