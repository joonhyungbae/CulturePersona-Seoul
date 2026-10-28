"""Wave metadata -- fielding period, recall window, sample design. None of it is in the
microdata.

When fieldwork happened, what period respondents were asked to recall, and how the
sample was drawn are absent from the raw data; they appear only in the body of the
reports. The values below were read from the report text, and each keeps the
source line number in the report's text extraction.

**The recall window does not coincide with the survey year, and that matters.** The
'2020' survey was fielded in January and February 2021 and asks about January to
December 2020 -- the whole of the pandemic year. 2015 is not a calendar year at all but
2014-06 to 2015-05.

The sampling, mode and frame description values below are transcriptions of the Korean
reports and are kept in their source wording.

Outputs:
  official_wave_metadata.csv  wave x fielding period x recall window x basis x source
  official_frame_design.csv   wave x frame x sampling method x mode x whether it
                              projects to a population
"""
from __future__ import annotations

import sys
from pathlib import Path

import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent))
import official_00_common as C

# (wave, fielding start, fielding end, recall start, recall end, basis, note, source)
WAVES = [
    (2014, "2014-12-02", "2014-12-26", "2013-12", "2014-11", "inferred",
     "the report does not state the recall window; inferred backwards from fieldwork "
     "in 2014-12",
     "data/interim/pdf_text/2014.txt:237"),
    (2015, "2015-08-24", "2015-09-23", "2014-06", "2015-05", "stated",
     "the recall window is stated in the question: June to May rather than a calendar "
     "year, ending three months before fieldwork",
     "data/interim/pdf_text/2015.txt:3473"),
    (2016, "2016-11-21", "2016-12-09", "2015-11", "2016-10", "inferred",
     "the report does not state the recall window; inferred backwards from fieldwork. "
     "The four samples were fielded over different periods (11-21 to 12-02, and to "
     "12-09)",
     "data/interim/pdf_text/2016.txt:346"),
    (2018, "2018-11-01", "2018-11-19", "2017-11", "2018-10", "inferred",
     "the culture-interested group was fielded 2018-11-01 to 11-15",
     "data/interim/md/2018_서울시민_문화향유_실태조사.md:176"),
    (2020, "2021-01-11", "2021-02-05", "2020-01", "2020-12", "stated",
     "the survey is named 2020 but was fielded in January and February 2021, so the "
     "recall window covers the whole pandemic year. The culture-interested group was "
     "fielded 2021-01-11 to 02-10",
     "data/interim/pdf_text/2020.txt:415"),
    (2022, "2022-11-22", "2022-12-30", "2021-11", "2022-10", "stated",
     "the disabled and multicultural samples were fielded 2022-11-28 to 12-30",
     "data/interim/pdf_text/2022.txt:408"),
    (2024, "2024-10-21", "2024-11-08", "2023-10", "2024-09", "stated",
     "the report gives 2024-11-01 to 11-31 for the culture-interested group, which is "
     "a typographical error -- November has no 31st. The disabled sample was fielded "
     "2024-11-14 to 12-30",
     "data/interim/pdf_text/2024_1.txt:247"),
]

# (wave, frame_code, sampling method, mode, description, source) -- transcribed
FRAME_DESIGN = [
    (2014, 1.0, "전수조사 (72,348명 이메일 발송)", "Web",
     "서울문화재단 홈페이지 회원 및 재단 사업 참여 경험자", "data/interim/pdf_text/2014.txt:230"),
    (2015, 9.0, "회원 리스트", "Web", "서울문화재단 홈페이지 회원", "data/interim/pdf_text/2015.txt:355"),
    (2015, 10.0, "패널 리스트", "Web", "문화정책패널(기존패널)", "data/interim/pdf_text/2015.txt:355"),
    (2016, 1.0, "온라인 패널", "Web", "서울 거주 온라인 조사 패널", "data/interim/pdf_text/2016.txt:752"),
    (2016, 2.0, "이용자 리스트", "Web", "서울시 문화뉴스레터 수신자", "data/interim/pdf_text/2016.txt:115"),
    (2016, 3.0, "회원 리스트", "Web", "서울시립미술관 회원", "data/interim/pdf_text/2016.txt:115"),
    (2016, 4.0, "회원 리스트", "Web", "서울문화재단 회원", "data/interim/pdf_text/2016.txt:115"),
    (2018, 1.0, "온라인 패널", "Web", "서울시민 온라인 패널",
     "data/interim/md/2018_서울시민_문화향유_실태조사.md:176"),
    (2020, 1.0, "온라인 패널", "Web", "패널 조사", "data/interim/pdf_text/2020.txt:415"),
    (2020, 4.0, "오프라인 모집", "면접", "오프라인 조사", "data/interim/pdf_text/2020.txt:415"),
    (2022, 1.0, "온라인 패널", "Web", "패널 조사", "data/interim/pdf_text/2022.txt:408"),
    (2022, 2.0, "이용자 리스트", "Web", "문화관심집단", "data/interim/pdf_text/2022.txt:408"),
    (2022, 3.0, "계통적표집", "면접", "면접 조사", "data/interim/pdf_text/2022.txt:408"),
    (2022, 4.0, "장애인 표본", "면접", "장애인", "data/interim/pdf_text/2022.txt:408"),
    (2022, 5.0, "다문화 표본", "면접", "다문화가정", "data/interim/pdf_text/2022.txt:408"),
    (2024, 1.0, "표본조사 + 레이크 가중", "Web·면접 병행", "서울시민 표본",
     "data/interim/pdf_text/2024_1.txt:247"),
    # Samples outside the general public. They do not project to a population, but
    # they were measured with the same instrument.
    (2018, 2.0, "기관 회원명부", "Web", "기관회원 (재단·기관 서비스 이용자)",
     "data/interim/md/2018_서울시민_문화향유_실태조사.md:176"),
    (2020, 2.0, "배너·이용자 모집", "Web", "문화관심집단",
     "data/interim/pdf_text/2020.txt:415"),
    (2024, 2.0, "이용자 리스트", "Web", "문화관심집단",
     "data/interim/pdf_text/2024_1.txt:247"),
    (2024, 3.0, "장애인 표본", "면접", "장애인",
     "data/interim/pdf_text/2024_1.txt:247"),
]


def main():
    W = pd.DataFrame(WAVES, columns=[
        "wave", "fielding_start", "fielding_end", "recall_start", "recall_end",
        "recall_basis", "note", "source"])
    D = pd.DataFrame(FRAME_DESIGN, columns=[
        "wave", "frame_code", "sampling", "mode", "frame_desc", "source"])

    frames = pd.read_csv(C.OUT / "official_sample_frames.csv")
    D = frames.merge(D, on=["wave", "frame_code"], how="left", validate="one_to_one")
    missing = D[D.sampling.isna()]
    assert not len(missing), \
        ("frames with no sample design description: "
         f"{missing[['wave','frame_code']].to_dict('records')}")

    W.to_csv(C.OUT / "official_wave_metadata.csv", index=False)
    D.to_csv(C.OUT / "official_frame_design.csv", index=False)

    print("== wave metadata ==")
    print(W[["wave", "fielding_start", "fielding_end", "recall_start", "recall_end",
             "recall_basis"]].to_string(index=False))
    print("\n== frame design ==")
    print(D[["wave", "frame_label", "n", "general_population", "sampling", "mode"]].to_string(index=False))
    print("\nnote: the 2020 survey was fielded in 2021-01 to 02; its recall window is "
          "the whole of 2020.")


if __name__ == "__main__":
    main()
