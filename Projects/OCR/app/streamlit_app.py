"""
app/streamlit_app.py
UI: загрузка PDF → пайплайн → просмотр и скачивание результата.

Запуск:
    streamlit run app/streamlit_app.py
"""
import io
import sys
import tempfile
from pathlib import Path

import streamlit as st
from PIL import Image

sys.path.insert(0, str(Path(__file__).parent.parent))
from pipeline.orchestrator import OCRPipeline


@st.cache_resource
def get_pipeline() -> OCRPipeline:
    return OCRPipeline()


def main():
    st.set_page_config(page_title="Оцифровка лекций", layout="wide")
    st.title("📄 Оцифровка рукописных лекций")
    st.caption("Загрузите PDF — получите оцифрованные страницы и Markdown.")

    # ── Загрузка файла ────────────────────────────────────────────────────────
    uploaded = st.file_uploader("Загрузите PDF", type=["pdf"])
    if not uploaded:
        st.info("Загрузите PDF файл чтобы начать.")
        return

    # ── Опции ─────────────────────────────────────────────────────────────────
    with st.expander("⚙️ Настройки"):
        process_all = st.checkbox("Все страницы", value=True)
        page_input  = st.text_input(
            "Страницы (через запятую, например: 1,2,3)",
            disabled=process_all
        )

    page_nums = None
    if not process_all and page_input.strip():
        try:
            page_nums = [int(p.strip()) for p in page_input.split(",")]
        except ValueError:
            st.error("Неверный формат страниц.")
            return

    # ── Запуск ────────────────────────────────────────────────────────────────
    if st.button("▶ Оцифровать", type="primary"):
        pipeline = get_pipeline()

        # Сохраняем загруженный PDF во временный файл
        with tempfile.NamedTemporaryFile(suffix=".pdf", delete=False) as f:
            f.write(uploaded.read())
            tmp_pdf = Path(f.name)

        # Прогресс
        progress_bar = st.progress(0.0)
        status_text  = st.empty()

        def on_progress(current: int, total: int, message: str):
            pct = current / max(total, 1)
            progress_bar.progress(pct)
            status_text.text(message)

        try:
            rendered_pages, full_md = pipeline.process(
                tmp_pdf, page_nums, progress_callback=on_progress
            )
        except Exception as e:
            st.error(f"Ошибка пайплайна: {e}")
            raise
        finally:
            tmp_pdf.unlink(missing_ok=True)

        progress_bar.progress(1.0)
        status_text.text("✅ Готово!")

        # ── Результаты ────────────────────────────────────────────────────────
        st.subheader("Результат")

        tab_pages, tab_md = st.tabs(["📄 Страницы", "📝 Markdown"])

        with tab_pages:
            for i, page_img in enumerate(rendered_pages):
                st.markdown(f"**Страница {i + 1}**")
                st.image(page_img, use_column_width=True)

                # Кнопка скачивания страницы
                buf = io.BytesIO()
                page_img.save(buf, format="PNG")
                st.download_button(
                    label=f"⬇ Скачать страницу {i+1} (PNG)",
                    data=buf.getvalue(),
                    file_name=f"page_{i+1:03d}.png",
                    mime="image/png",
                    key=f"dl_page_{i}",
                )

        with tab_md:
            st.markdown(full_md, unsafe_allow_html=True)
            st.download_button(
                label="⬇ Скачать Markdown",
                data=full_md.encode("utf-8"),
                file_name="lecture_ocr.md",
                mime="text/markdown",
            )


if __name__ == "__main__":
    main()
