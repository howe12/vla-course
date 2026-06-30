#!/usr/bin/env python3
"""Export VLA course to clean Markdown. v4 — video links, cleaner output."""

import re, subprocess, os

BASE_RAW = "https://raw.githubusercontent.com/howe12/vla-course/main"

with open("/develop/vla-course/index.html", "r") as f:
    html = f.read()

chapters = [
    ("1", "VLA 基础与架构", "视觉-语言-动作模型的核心概念、技术演进与系统架构。"),
    ("2", "仿真环境搭建", "MuJoCo + LIBERO 仿真平台，为 VLA 实验准备沙盒。"),
    ("3", "L40 云端训练环境", "SSH 连接、CUDA 配置、依赖管理——为后续训练备好算力。"),
    ("4", "OpenVLA 实战", "7B 开源 VLA 模型的推理、适配与理解。"),
    ("5", "SmolVLA 轻量 VLA", "4.5 亿参数，单卡训练，CPU 推理——最亲民的 VLA。"),
    ("6", "Pi0 扩散 VLA", "基于扩散模型的 SOTA VLA，理解动作生成的另一种范式。"),
    ("7", "Gemini 真机部署", "将云端训练的 VLA 模型部署到双子座机器人上验证。"),
    ("8", "VLA 前沿与竞赛", "ACoT-VLA、ManipDojo、ICRA2026——站在最前沿。"),
]

def extract_chapter_content(html, ch_num):
    pattern = f"function r_ch{ch_num}(area,ch)"
    idx = html.find(pattern)
    if idx < 0: return None
    inner_start = html.find("area.innerHTML=", idx)
    if inner_start < 0: return None
    body_start = html.index("\n", inner_start) + 1
    end_marker = html.find("';\n}\n", body_start)
    if end_marker < 0:
        end_marker = html.find("';\n}", body_start)
    if end_marker < 0: return None
    
    raw = html[body_start:end_marker+2]
    lines = raw.split('\n')
    result = []
    for line in lines:
        stripped = line.strip()
        if not stripped: continue
        if "'" not in stripped: continue
        first_q = stripped.index("'")
        last_q = stripped.rfind("'")
        if last_q > first_q:
            result.append(stripped[first_q+1:last_q])
    
    content = '\n'.join(result)
    content = content.replace("\\n", "\n")
    content = content.replace("\\'", "'")
    content = content.replace('\\"', '"')
    content = content.replace("\\\\", "\\")
    return content


def resolve_template_vars(content, ch_num, ch_title, ch_desc):
    content = content.replace("'+ch.num+'", ch_num)
    content = content.replace("'+ch.title+'", ch_title)
    content = content.replace("'+ch.desc+'", ch_desc)
    return content


def fix_media_urls(content):
    # Images: figs/ → absolute
    content = re.sub(r'src="figs/([^"]+)"', rf'src="{BASE_RAW}/figs/\1"', content)
    # Images: raw.githubusercontent already
    # Videos: convert <video src="..."> to markdown links before pandoc
    content = re.sub(
        r'<video src="([^"]+)"[^>]*></video>',
        r'<a href="\1">🎬 点击播放视频</a>',
        content
    )
    # Also handle relative video paths
    content = re.sub(
        r'<a href="videos/([^"]+)">',
        rf'<a href="{BASE_RAW}/videos/\1">',
        content
    )
    return content


def preprocess_html(content):
    """Clean up HTML for pandoc."""
    # Strip ch-header div (redundant with our markdown header)
    content = re.sub(r'<div class="ch-header">.*?</div>', '', content, flags=re.DOTALL)
    # Convert callout boxes to blockquotes
    content = re.sub(
        r'<div class="callout tip"><div class="lb">([^<]+)</div><ul',
        r'<blockquote><p><strong>\1</strong></p><ul', content
    )
    content = re.sub(
        r'<div class="callout"><div class="lb">([^<]+)</div><p[^>]*>',
        r'<blockquote><p><strong>\1</strong></p><p>', content
    )
    content = re.sub(
        r'<div class="callout warn"><div class="lb">([^<]+)</div><p[^>]*>',
        r'<blockquote><p><strong>⚠️ \1</strong></p><p>', content
    )
    # Convert card to blockquote
    content = re.sub(
        r'<div class="card"><div class="card-label">([^<]+)</div>',
        r'<blockquote><p><strong>\1</strong></p>', content
    )
    # Convert practice-box
    content = re.sub(
        r'<div class="practice-box"><div class="pb-title">([^<]+)</div>',
        r'<blockquote><p><strong>🧪 \1</strong></p>', content
    )
    # Convert transition-note
    content = re.sub(
        r'<div class="transition-note">(.*?)</div>',
        r'<blockquote>\1</blockquote>', content, flags=re.DOTALL
    )
    # Close divs that become blockquotes
    content = content.replace('</div>', '</blockquote>')
    # But don't double-close — clean up
    content = content.replace('</blockquote></blockquote>', '</blockquote>')
    # Convert section to div
    content = content.replace('<section>', '<div>').replace('</section>', '</div>')
    # Make details/summary work
    content = content.replace('<details>', '<p><details>').replace('</details>', '</details></p>')
    
    return content


def postprocess_markdown(md):
    """Clean up pandoc output."""
    # Remove pandoc div syntax
    md = re.sub(r'::: {[^}]+}', '', md)
    md = re.sub(r':::', '', md)
    # Fix excessive blank lines
    md = re.sub(r'\n{4,}', '\n\n\n', md)
    # Fix escaped quotes
    md = md.replace('\\"', '"').replace("\\'", "'")
    # Fix stray closing blockquote tags
    md = md.replace('\n</blockquote>', '')
    md = md.replace('</blockquote>', '')
    # Fix empty blockquote lines
    md = re.sub(r'^> \s*$', '', md, flags=re.MULTILINE)
    # Make video links more visible  
    md = md.replace('[🎬 点击播放视频]', '🎬 [点击播放视频]')
    
    return md


def html_to_markdown(html_content):
    full_html = f"<!DOCTYPE html>\n<html><head><meta charset=\"utf-8\"></head><body>\n{html_content}\n</body></html>"
    result = subprocess.run(
        ["pandoc", "-f", "html", "-t", "markdown", "--wrap=none"],
        input=full_html, capture_output=True, text=True, timeout=120
    )
    return postprocess_markdown(result.stdout)


output_dir = "/develop/vla-course/docs/md"
os.makedirs(output_dir, exist_ok=True)
all_parts = []

for ch_num, ch_title, ch_desc in chapters:
    print(f"Chapter {ch_num}: {ch_title}...", end=" ")
    content = extract_chapter_content(html, ch_num)
    if not content:
        print("FAILED")
        continue
    
    content = resolve_template_vars(content, ch_num, ch_title, ch_desc)
    content = fix_media_urls(content)
    content = preprocess_html(content)
    md = html_to_markdown(content)
    
    header = f"# 第 {ch_num} 章：{ch_title}\n\n> {ch_desc}\n\n---\n\n"
    ch_path = os.path.join(output_dir, f"chapter_{ch_num}.md")
    with open(ch_path, "w") as f:
        f.write(header + md)
    
    all_parts.append(header + md)
    img_count = md.count('![')
    vid_count = md.count('点击播放视频') + md.count('.mp4')
    print(f"OK ({len(md)} chars, {img_count} imgs, {vid_count} vids)")

with open(os.path.join(output_dir, "vla-course-full.md"), "w") as f:
    f.write("# VLA 从零到部署 — 具身智能教科书\n\n")
    f.write("> 具身智能 · MuJoCo + LeRobot + L40 + Gemini\n\n---\n\n")
    f.write("\n\n---\n\n".join(all_parts))

total = sum(len(p) for p in all_parts)
print(f"\nDone! → {output_dir}/")
print(f"  chapter_1~8.md + vla-course-full.md (~{total//1000}K total)")
