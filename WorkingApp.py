import joblib
import numpy as np
import pandas as pd
import streamlit as st


MODEL_PATH = "salary_model.joblib"
DATA_PATH = "ds_salaries.csv"

VALID_EXPERIENCE_LEVELS = ["EN", "MI", "SE", "EX"]
VALID_EMPLOYMENT_TYPES = ["PT", "FT", "CT", "FL"]
VALID_COMPANY_SIZES = ["S", "M", "L"]
VALID_REMOTE_RATIOS = [0, 50, 100]

EXPERIENCE_LEVEL_LABELS = {
	"EN": "EN - Entry-level",
	"MI": "MI - Mid-level",
	"SE": "SE - Senior-level",
	"EX": "EX - Executive-level",
}

EMPLOYMENT_TYPE_LABELS = {
	"PT": "PT - Part-time",
	"FT": "FT - Full-time",
	"CT": "CT - Contract",
	"FL": "FL - Freelance",
}

COMPANY_SIZE_LABELS = {
	"S": "S - Small",
	"M": "M - Medium",
	"L": "L - Large",
}

COUNTRY_CODE_LABELS = {
	"AE": "United Arab Emirates",
	"AR": "Argentina",
	"AS": "American Samoa",
	"AT": "Austria",
	"AU": "Australia",
	"BE": "Belgium",
	"BO": "Bolivia",
	"BR": "Brazil",
	"CA": "Canada",
	"CF": "Central African Republic",
	"CH": "Switzerland",
	"CL": "Chile",
	"CN": "China",
	"CO": "Colombia",
	"CR": "Costa Rica",
	"CZ": "Czech Republic",
	"DE": "Germany",
	"DK": "Denmark",
	"DZ": "Algeria",
	"EE": "Estonia",
	"EG": "Egypt",
	"ES": "Spain",
	"FI": "Finland",
	"FR": "France",
	"GB": "United Kingdom",
	"GH": "Ghana",
	"GR": "Greece",
	"HK": "Hong Kong",
	"HN": "Honduras",
	"HR": "Croatia",
	"HU": "Hungary",
	"IE": "Ireland",
	"IL": "Israel",
	"IN": "India",
	"IQ": "Iraq",
	"IR": "Iran",
	"IT": "Italy",
	"JE": "Jersey",
	"JP": "Japan",
	"KE": "Kenya",
	"LU": "Luxembourg",
	"MD": "Moldova",
	"MT": "Malta",
	"MU": "Mauritius",
	"MX": "Mexico",
	"MY": "Malaysia",
	"NG": "Nigeria",
	"NL": "Netherlands",
	"NZ": "New Zealand",
	"PH": "Philippines",
	"PK": "Pakistan",
	"PL": "Poland",
	"PR": "Puerto Rico",
	"PT": "Portugal",
	"RO": "Romania",
	"RS": "Serbia",
	"RU": "Russia",
	"SG": "Singapore",
	"SI": "Slovenia",
	"TN": "Tunisia",
	"TR": "Turkey",
	"UA": "Ukraine",
	"US": "United States",
	"VN": "Vietnam",
}


@st.cache_resource
def load_model(path: str):
	return joblib.load(path)


@st.cache_data
def load_dropdown_options(path: str) -> dict:
	default_options = {
		"work_year": [2023],
		"job_title": ["Data Scientist"],
		"employee_residence": ["US"],
		"company_location": ["US"],
	}

	try:
		df = pd.read_csv(path)
	except Exception:
		return default_options

	def _safe_unique(col: str, uppercase: bool = False) -> list[str]:
		if col not in df.columns:
			return []
		series = df[col].dropna().astype(str).str.strip()
		if uppercase:
			series = series.str.upper()
		values = sorted(series[series != ""].unique().tolist())
		return values

	work_years: list[int] = []
	if "work_year" in df.columns:
		work_year_series = pd.to_numeric(df["work_year"], errors="coerce").dropna().astype(int)
		work_years = sorted(work_year_series.unique().tolist())

	return {
		"work_year": work_years or default_options["work_year"],
		"job_title": _safe_unique("job_title") or default_options["job_title"],
		"employee_residence": _safe_unique("employee_residence", uppercase=True)
		or default_options["employee_residence"],
		"company_location": _safe_unique("company_location", uppercase=True)
		or default_options["company_location"],
	}


def _walk_estimators(obj, seen: set[int] | None = None):
	if seen is None:
		seen = set()

	obj_id = id(obj)
	if obj_id in seen:
		return
	seen.add(obj_id)

	yield obj

	if hasattr(obj, "steps"):
		for _, step in getattr(obj, "steps"):
			yield from _walk_estimators(step, seen)

	if hasattr(obj, "named_steps"):
		for step in getattr(obj, "named_steps").values():
			yield from _walk_estimators(step, seen)

	if hasattr(obj, "transformers"):
		for _, transformer, _ in getattr(obj, "transformers"):
			if transformer not in ("drop", "passthrough"):
				yield from _walk_estimators(transformer, seen)

	if hasattr(obj, "transformers_"):
		for _, transformer, _ in getattr(obj, "transformers_"):
			if transformer not in ("drop", "passthrough"):
				yield from _walk_estimators(transformer, seen)

	if hasattr(obj, "transformer_list"):
		for _, transformer in getattr(obj, "transformer_list"):
			yield from _walk_estimators(transformer, seen)

	for attr_name in ("estimator", "base_estimator", "regressor", "classifier"):
		if hasattr(obj, attr_name):
			yield from _walk_estimators(getattr(obj, attr_name), seen)


def patch_simple_imputer_compat(model) -> int:
	patched_count = 0
	for estimator in _walk_estimators(model):
		if estimator.__class__.__name__ == "SimpleImputer" and not hasattr(estimator, "_fill_dtype"):
			if hasattr(estimator, "statistics_") and getattr(estimator, "statistics_") is not None:
				estimator._fill_dtype = getattr(estimator, "statistics_").dtype
			else:
				estimator._fill_dtype = np.dtype("O")
			patched_count += 1
	return patched_count


def predict_salary(model, sample: dict) -> float:
	input_df = pd.DataFrame([sample])
	try:
		prediction = model.predict(input_df)
	except AttributeError as exc:
		if "_fill_dtype" not in str(exc):
			raise
		patch_simple_imputer_compat(model)
		prediction = model.predict(input_df)
	return float(prediction[0])


def format_country_code(code: str) -> str:
	upper_code = str(code).upper()
	country_name = COUNTRY_CODE_LABELS.get(upper_code)
	if country_name:
		return f"{upper_code} - {country_name}"
	return upper_code


def filter_known_country_codes(codes: list[str]) -> list[str]:
	known_codes = [code for code in codes if code in COUNTRY_CODE_LABELS]
	return known_codes or ["US"]


def main() -> None:
	st.set_page_config(page_title="Salary Predictor", layout="centered")
	st.markdown("## Data Science Salary Predictor")
	st.caption(
		"Estimate annual compensation for data science roles using key job and location details."
	)

	with st.sidebar:
		st.markdown("### About this app")
		st.write(
			"This Data Science Salary Predictor uses a Linear Regression model (R² ≈ 0.83) trained on real-world "
        	"salary datasets to estimate compensation for various roles.\n\n"
        	"Users can input details such as experience level, job title, company size, and work setting to receive "
        	"an instant salary prediction.\n\n"
        	"This project demonstrates an end-to-end machine learning pipeline, including data preprocessing, "
        	"model training, evaluation, and deployment using Streamlit."
		)


	try:
		model = load_model(MODEL_PATH)
		patched = patch_simple_imputer_compat(model)
	except FileNotFoundError:
		st.error(f"Model file not found: {MODEL_PATH}")
		st.stop()
	except Exception as exc:
		st.error(f"Could not load model: {exc}")
		st.stop()

	dropdown_options = load_dropdown_options(DATA_PATH)
	work_year_options = dropdown_options["work_year"]
	job_title_options = dropdown_options["job_title"]
	employee_residence_options = filter_known_country_codes(dropdown_options["employee_residence"])
	company_location_options = filter_known_country_codes(dropdown_options["company_location"])

	st.info("Model: Linear Regression | Performance: R² ≈ 0.83")

	with st.expander("How it works"):
		st.write(
			"The model uses supervised learning (Linear Regression) trained on historical "
			"data science salary records. It estimates salary from the selected features: "
			"work year, experience level, employment type, job title, employee residence, "
			"remote ratio, company location, and company size."
		)




	with st.form("prediction_form"):
		dataset_min_year = int(min(work_year_options))
		dataset_max_year = int(max(work_year_options))
		min_selectable_year = dataset_min_year - 20
		max_selectable_year = dataset_max_year + 20
		work_year_default = 2023 if 2023 in work_year_options else int(work_year_options[0])
		left_col, right_col = st.columns(2)

		with left_col:
			work_year = st.number_input(
				"Work year",
				min_value=min_selectable_year,
				max_value=max_selectable_year,
				value=work_year_default,
				step=1,
				help=(
					f"Use + or - to choose a year from {min_selectable_year} to {max_selectable_year} "
					f"(dataset range {dataset_min_year}-{dataset_max_year}, extended by +/- 20 years)."
				),
			)
			experience_level = st.selectbox(
				"Experience level",
				VALID_EXPERIENCE_LEVELS,
				index=2,
				format_func=lambda code: EXPERIENCE_LEVEL_LABELS.get(code, code),
			)
			employment_type = st.selectbox(
				"Employment type",
				VALID_EMPLOYMENT_TYPES,
				index=1,
				format_func=lambda code: EMPLOYMENT_TYPE_LABELS.get(code, code),
			)

		job_title_default_index = job_title_options.index("Data Scientist") if "Data Scientist" in job_title_options else 0
		employee_residence_default_index = (
			employee_residence_options.index("US") if "US" in employee_residence_options else 0
		)
		company_location_default_index = company_location_options.index("US") if "US" in company_location_options else 0

		with right_col:
			job_title = st.selectbox("Job title", job_title_options, index=job_title_default_index)
			employee_residence = st.selectbox(
				"Employee residence (country code)",
				employee_residence_options,
				index=employee_residence_default_index,
				format_func=format_country_code,
			)
			remote_ratio = st.selectbox("Remote ratio", VALID_REMOTE_RATIOS, index=2)
			company_location = st.selectbox(
				"Company location (country code)",
				company_location_options,
				index=company_location_default_index,
				format_func=format_country_code,
			)
			company_size = st.selectbox(
				"Company size",
				VALID_COMPANY_SIZES,
				index=1,
				format_func=lambda code: COMPANY_SIZE_LABELS.get(code, code),
			)

		submitted = st.form_submit_button("Predict salary")


	if submitted:
		sample = {
			"work_year": int(work_year),
			"experience_level": experience_level,
			"employment_type": employment_type,
			"job_title": job_title,
			"employee_residence": employee_residence,
			"remote_ratio": int(remote_ratio),
			"company_location": company_location,
			"company_size": company_size,
		}

		try:
			predicted_salary = predict_salary(model, sample)
			st.success(f"Predicted salary: ${predicted_salary:,.2f}")
			st.dataframe(pd.DataFrame([sample]), use_container_width=True)
		except Exception as exc:
			st.error(f"Prediction failed: {exc}")

	st.markdown("---")
	st.caption("Built with Streamlit | By Group 4 - Data Science Project (S4 IT 2026 GECBH)")


if __name__ == "__main__":
	main()
