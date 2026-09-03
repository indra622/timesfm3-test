# TimesFM 3 연구 문서

현재 논문 버전은 **v11**이다. v11은 9.5절을 실험에서 확인된 세 가지 이득 경로, 보험 업무에서 다변량 예측이 필요한 사례 7개와 단변량 대비 우위를 기대하는 이유, 부적합 영역, 검증 프로토콜로 다시 구성한 개정판이며, 수치 결과와 결론은 v08 이후 같다.

## 현재 버전

- [논문형 연구리포트 v11](<./TimesFM 3 다변량 예측 실증 연구리포트 v11.md>)
- [논문형 연구리포트 v11 PDF](<./TimesFM 3 다변량 예측 실증 연구리포트 v11.pdf>)
- [학술대회 발표 슬라이드 v2 (HTML)](<./slides/TimesFM 3 다변량 예측 실증 발표 v2.html>) / [PDF](<./slides/TimesFM 3 다변량 예측 실증 발표 v2.pdf>)

## 이전 버전

- [논문형 연구리포트 v10](<./TimesFM 3 다변량 예측 실증 연구리포트 v10.md>) / [PDF](<./TimesFM 3 다변량 예측 실증 연구리포트 v10.pdf>) — 보험 응용 함의 절 추가
- [논문형 연구리포트 v09](<./TimesFM 3 다변량 예측 실증 연구리포트 v09.md>) / [PDF](<./TimesFM 3 다변량 예측 실증 연구리포트 v09.pdf>) — 합성 데이터 절(7.1) 재서술
- [논문형 연구리포트 v08](<./TimesFM 3 다변량 예측 실증 연구리포트 v08.md>) / [PDF](<./TimesFM 3 다변량 예측 실증 연구리포트 v08.pdf>)
- [발표 슬라이드 v1](<./slides/TimesFM 3 다변량 예측 실증 발표 v1.html>) / [PDF](<./slides/TimesFM 3 다변량 예측 실증 발표 v1.pdf>)

PDF는 `uv run --with mdit-py-plugins python scripts/build_paper_pdf.py "docs/<paper>.md" "docs/<paper>.pdf" /tmp/paper.html`로 다시 만들 수 있다(headless Chrome, KaTeX CDN 필요).

그림 자산은 [`assets/`](./assets/)에 보관한다. 보고서의 로컬 결과 링크는 프로젝트 루트의 `artifacts/`를 기준으로 한다.
