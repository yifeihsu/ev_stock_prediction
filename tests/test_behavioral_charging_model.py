import sys
import subprocess
import tempfile
import unittest
from pathlib import Path
import json

import pandas as pd


ROOT = Path(__file__).resolve().parents[1]
if str(ROOT / "scripts") not in sys.path:
    sys.path.insert(0, str(ROOT / "scripts"))

import build_behavioral_charging_model as bcm  # noqa: E402


def trip_row(**overrides):
    row = {
        "SYN_RECORD_ID": "ALBANY00000001",
        "home_geoid": "360010001001",
        "PERSONID": 1,
        "HHS": 2,
        "VEH": 1,
        "HHI": "MI",
        "URBRUR": 1.0,
        "HOMEOWN": 1.0,
        "TDTRPNUM": 1,
        "STRTTIME": 800,
        "ENDTIME": 830,
        "TRVLCMIN": 30,
        "TRPMILES": 10.0,
        "DWELTIME": 60,
        "WHYTRP1S": 1,
        "HHSTFIPS": 36,
        "PUMA": 3602002,
        "CITYTOWN_NAME": "Albany",
        "HOME_X": -73.75,
        "HOME_Y": 42.65,
        "ACTIVITY_X": -73.75,
        "ACTIVITY_Y": 42.65,
        "ACTIVITY_ADDR": "home",
        "ACTIVITY_TYPE": "HOME",
        "START_X": -73.70,
        "START_Y": 42.70,
        "SHORTEST_TDIST_miles": 10.0,
        "SHORTEST_PDIST_miles": 9.0,
        "source_row_id": 0,
    }
    row.update(overrides)
    return row


class BehavioralChargingModelTests(unittest.TestCase):
    def test_parse_hhmm_to_minutes(self):
        self.assertEqual(bcm.parse_hhmm_to_minutes(737), 457.0)
        self.assertEqual(bcm.parse_hhmm_to_minutes(0), 0.0)
        self.assertTrue(pd.isna(bcm.parse_hhmm_to_minutes(2460)))
        self.assertTrue(pd.isna(bcm.parse_hhmm_to_minutes("bad")))

    def test_network_distance_prefers_routed_fields(self):
        df = pd.DataFrame(
            [
                {
                    "SHORTEST_TDIST_miles": 12.0,
                    "SHORTEST_PDIST_miles": 9.0,
                    "TRPMILES": 4.0,
                },
                {
                    "SHORTEST_TDIST_miles": 0.0,
                    "SHORTEST_PDIST_miles": 8.0,
                    "TRPMILES": 4.0,
                },
                {
                    "SHORTEST_TDIST_miles": 0.0,
                    "SHORTEST_PDIST_miles": 0.0,
                    "TRPMILES": 4.0,
                },
                {
                    "SHORTEST_TDIST_miles": 0.0,
                    "SHORTEST_PDIST_miles": 0.0,
                    "TRPMILES": 999.0,
                },
                {
                    "SHORTEST_TDIST_miles": 0.0,
                    "SHORTEST_PDIST_miles": 0.0,
                    "TRPMILES": 4.0,
                    "straight_line_miles": 20.0,
                },
                {
                    "SHORTEST_TDIST_miles": 4.0,
                    "SHORTEST_PDIST_miles": 3.0,
                    "TRPMILES": 999.0,
                    "straight_line_miles": 20.0,
                },
            ]
        )
        route_miles = bcm.choose_network_miles(
            df,
            max_fallback_miles=500,
            distance_source="route",
        ).tolist()
        trip_miles = bcm.choose_network_miles(
            df,
            max_fallback_miles=500,
            distance_source="trip",
        ).tolist()
        self.assertEqual(route_miles[:3], [12.0, 8.0, 4.0])
        self.assertEqual(trip_miles[:3], [4.0, 4.0, 4.0])
        self.assertTrue(pd.isna(route_miles[3]))
        self.assertTrue(pd.isna(route_miles[4]))
        self.assertTrue(pd.isna(route_miles[5]))

    def test_reconstruct_vehicle_day_replaces_final_negative_dwell(self):
        raw = pd.DataFrame(
            [
                trip_row(
                    TDTRPNUM=1,
                    STRTTIME=800,
                    ENDTIME=830,
                    DWELTIME=60,
                    ACTIVITY_TYPE="WORK",
                    WHYTRP1S=10,
                    ACTIVITY_X=-73.70,
                    ACTIVITY_Y=42.70,
                    START_X=-73.75,
                    START_Y=42.65,
                    source_row_id=0,
                ),
                trip_row(
                    TDTRPNUM=2,
                    STRTTIME=930,
                    ENDTIME=1000,
                    DWELTIME=-9,
                    ACTIVITY_TYPE="HOME",
                    WHYTRP1S=1,
                    ACTIVITY_X=-73.75,
                    ACTIVITY_Y=42.65,
                    START_X=-73.70,
                    START_Y=42.70,
                    source_row_id=1,
                ),
            ]
        )
        trips, qa = bcm.reconstruct_vehicle_day_trips(
            raw,
            max_fallback_miles=500,
            distance_source="route",
            origin_mode="reconstructed",
            coordinate_tolerance=0.001,
        )
        self.assertEqual(qa["vehicle_day_count"], 1)
        self.assertEqual(trips["vehicle_day_id"].nunique(), 1)
        final = trips.sort_values("TDTRPNUM").iloc[-1]
        self.assertEqual(final["dwell_min"], 1320.0)

    def test_midnight_trip_sequence_does_not_create_new_vehicle_day(self):
        raw = pd.DataFrame(
            [
                trip_row(
                    TDTRPNUM=1,
                    STRTTIME=2225,
                    ENDTIME=2235,
                    DWELTIME=95,
                    ACTIVITY_TYPE="WORK",
                    WHYTRP1S=10,
                    ACTIVITY_X=-73.70,
                    ACTIVITY_Y=42.70,
                    START_X=-73.75,
                    START_Y=42.65,
                    source_row_id=0,
                ),
                trip_row(
                    TDTRPNUM=2,
                    STRTTIME=10,
                    ENDTIME=20,
                    DWELTIME=-9,
                    ACTIVITY_TYPE="HOME",
                    WHYTRP1S=1,
                    ACTIVITY_X=-73.75,
                    ACTIVITY_Y=42.65,
                    START_X=-73.70,
                    START_Y=42.70,
                    source_row_id=1,
                ),
            ]
        )
        trips, qa = bcm.reconstruct_vehicle_day_trips(
            raw,
            max_fallback_miles=500,
            distance_source="route",
            origin_mode="reconstructed",
            coordinate_tolerance=0.001,
        )
        self.assertEqual(qa["chain_breaks_impossible_time"], 0)
        self.assertEqual(qa["vehicle_day_count"], 1)
        self.assertEqual(trips.sort_values("TDTRPNUM")["start_time_min"].tolist(), [1345.0, 1450.0])

    def test_trip_number_gap_after_distance_drop_starts_new_chain(self):
        raw = pd.DataFrame(
            [
                trip_row(
                    TDTRPNUM=1,
                    STRTTIME=800,
                    ENDTIME=810,
                    ACTIVITY_TYPE="WORK",
                    WHYTRP1S=10,
                    ACTIVITY_X=-73.70,
                    ACTIVITY_Y=42.70,
                    SHORTEST_TDIST_miles=10.0,
                    source_row_id=0,
                ),
                trip_row(
                    TDTRPNUM=2,
                    STRTTIME=820,
                    ENDTIME=830,
                    ACTIVITY_TYPE="SHOPPING/ERRANDS",
                    WHYTRP1S=40,
                    ACTIVITY_X=-73.71,
                    ACTIVITY_Y=42.71,
                    SHORTEST_TDIST_miles=0.0,
                    SHORTEST_PDIST_miles=0.0,
                    TRPMILES=999.0,
                    source_row_id=1,
                ),
                trip_row(
                    TDTRPNUM=3,
                    STRTTIME=900,
                    ENDTIME=910,
                    ACTIVITY_TYPE="HOME",
                    WHYTRP1S=1,
                    ACTIVITY_X=-73.75,
                    ACTIVITY_Y=42.65,
                    SHORTEST_TDIST_miles=10.0,
                    source_row_id=2,
                ),
            ]
        )
        trips, qa = bcm.reconstruct_vehicle_day_trips(
            raw,
            max_fallback_miles=500,
            distance_source="route",
            origin_mode="reconstructed",
            coordinate_tolerance=0.001,
        )
        self.assertEqual(len(trips), 2)
        self.assertEqual(qa["invalid_distance_rows_dropped"], 1)
        self.assertEqual(qa["chain_breaks_trip_number_gap"], 1)
        self.assertEqual(qa["vehicle_day_count"], 2)

    def test_first_row_after_trip_gap_uses_input_origin_not_previous_kept_destination(self):
        raw = pd.DataFrame(
            [
                trip_row(
                    TDTRPNUM=1,
                    STRTTIME=800,
                    ENDTIME=810,
                    ACTIVITY_TYPE="WORK",
                    WHYTRP1S=10,
                    ACTIVITY_X=-73.70,
                    ACTIVITY_Y=42.70,
                    START_X=-73.75,
                    START_Y=42.65,
                    SHORTEST_TDIST_miles=10.0,
                    source_row_id=0,
                ),
                trip_row(
                    TDTRPNUM=2,
                    STRTTIME=820,
                    ENDTIME=830,
                    ACTIVITY_TYPE="SHOPPING/ERRANDS",
                    WHYTRP1S=40,
                    ACTIVITY_X=-73.71,
                    ACTIVITY_Y=42.71,
                    START_X=-73.70,
                    START_Y=42.70,
                    SHORTEST_TDIST_miles=0.0,
                    SHORTEST_PDIST_miles=0.0,
                    TRPMILES=999.0,
                    source_row_id=1,
                ),
                trip_row(
                    TDTRPNUM=3,
                    STRTTIME=900,
                    ENDTIME=910,
                    ACTIVITY_TYPE="SOCIAL/REC",
                    WHYTRP1S=50,
                    ACTIVITY_X=-73.61,
                    ACTIVITY_Y=42.61,
                    START_X=-73.60,
                    START_Y=42.60,
                    SHORTEST_TDIST_miles=15.0,
                    source_row_id=2,
                ),
            ]
        )
        trips, qa = bcm.reconstruct_vehicle_day_trips(
            raw,
            max_fallback_miles=500,
            distance_source="route",
            origin_mode="reconstructed",
            coordinate_tolerance=0.001,
        )
        gap_row = trips.loc[trips["TDTRPNUM"].eq(3)].iloc[0]
        self.assertEqual(qa["chain_breaks_trip_number_gap"], 1)
        self.assertEqual(gap_row["origin_lon"], gap_row["input_origin_lon"])
        self.assertEqual(gap_row["origin_lat"], gap_row["input_origin_lat"])
        self.assertEqual(gap_row["origin_lon"], -73.60)
        self.assertEqual(gap_row["origin_lat"], 42.60)

    def test_raw_dwell_mismatch_is_reported_and_inferred_gap_used(self):
        raw = pd.DataFrame(
            [
                trip_row(
                    TDTRPNUM=1,
                    STRTTIME=800,
                    ENDTIME=830,
                    DWELTIME=10,
                    ACTIVITY_TYPE="WORK",
                    WHYTRP1S=10,
                    ACTIVITY_X=-73.70,
                    ACTIVITY_Y=42.70,
                    source_row_id=0,
                ),
                trip_row(
                    TDTRPNUM=2,
                    STRTTIME=930,
                    ENDTIME=1000,
                    DWELTIME=-9,
                    ACTIVITY_TYPE="HOME",
                    WHYTRP1S=1,
                    ACTIVITY_X=-73.75,
                    ACTIVITY_Y=42.65,
                    source_row_id=1,
                ),
            ]
        )
        trips, qa = bcm.reconstruct_vehicle_day_trips(
            raw,
            max_fallback_miles=500,
            distance_source="route",
            origin_mode="reconstructed",
            coordinate_tolerance=0.001,
        )
        first = trips.sort_values("TDTRPNUM").iloc[0]
        self.assertEqual(first["dwell_min"], 60.0)
        self.assertEqual(qa["raw_dwell_gap_mismatch_rows"], 1)
        self.assertEqual(qa["max_abs_dwell_gap_error_min"], 50.0)

    def test_charging_allocation_unmanaged_and_managed_deliver_same_energy(self):
        raw = pd.DataFrame(
            [
                trip_row(
                    TDTRPNUM=1,
                    STRTTIME=800,
                    ENDTIME=830,
                    DWELTIME=-9,
                    ACTIVITY_TYPE="HOME",
                    WHYTRP1S=1,
                    source_row_id=0,
                    SHORTEST_TDIST_miles=10.0,
                )
            ]
        )
        trips, _ = bcm.reconstruct_vehicle_day_trips(
            raw,
            max_fallback_miles=500,
            distance_source="route",
            origin_mode="reconstructed",
            coordinate_tolerance=0.001,
        )
        stops = bcm.classify_stop_opportunities(
            trips,
            availability_scenario="base",
            availability_choice_model="gridup",
            charger_assumptions=bcm.load_charger_assumptions(),
            min_dwell_min=20,
            long_public_dwell_min=90,
        )
        unmanaged, _ = bcm.build_charging_events(
            stops,
            scenario_id="base_unmanaged",
            managed=False,
            efficiency_kwh_per_mile=0.42,
        )
        managed, _ = bcm.build_charging_events(
            stops,
            scenario_id="base_managed",
            managed=True,
            efficiency_kwh_per_mile=0.42,
        )
        self.assertAlmostEqual(unmanaged["energy_delivered_kwh"].sum(), 4.2 * 0.5)
        self.assertAlmostEqual(managed["energy_delivered_kwh"].sum(), 4.2 * 0.5)
        self.assertGreater(managed["event_duration_min"].iloc[0], unmanaged["event_duration_min"].iloc[0])
        self.assertLess(managed["expected_power_kw"].iloc[0], unmanaged["expected_power_kw"].iloc[0])

    def test_gridup_availability_increases_below_peak_stopped_count(self):
        stops = pd.DataFrame(
            [
                {
                    "charge_location_type": "work",
                    "charger_assumption_key": "work",
                    "charger_type": "work_l2",
                    "destination_lon": -73.75,
                    "destination_lat": 42.65,
                    "arrival_time_min": 8 * 60,
                    "departure_time_min": 8 * 60 + 30,
                    "dwell_min": 30,
                    "rated_power_kw": 7.2,
                    "peak_demand_hour_probability": 0.5,
                },
                {
                    "charge_location_type": "work",
                    "charger_assumption_key": "work",
                    "charger_type": "work_l2",
                    "destination_lon": -73.75,
                    "destination_lat": 42.65,
                    "arrival_time_min": 8 * 60 + 15,
                    "departure_time_min": 8 * 60 + 45,
                    "dwell_min": 30,
                    "rated_power_kw": 7.2,
                    "peak_demand_hour_probability": 0.5,
                },
                {
                    "charge_location_type": "work",
                    "charger_assumption_key": "work",
                    "charger_type": "work_l2",
                    "destination_lon": -73.75,
                    "destination_lat": 42.65,
                    "arrival_time_min": 9 * 60,
                    "departure_time_min": 9 * 60 + 30,
                    "dwell_min": 30,
                    "rated_power_kw": 7.2,
                    "peak_demand_hour_probability": 0.5,
                },
            ]
        )
        availability = bcm.gridup_time_varying_availability(
            stops,
            peak_probability_col="peak_demand_hour_probability",
        )
        self.assertAlmostEqual(availability.iloc[0], 0.5)
        self.assertAlmostEqual(availability.iloc[1], 0.5)
        self.assertAlmostEqual(availability.iloc[2], 1.0)

    def test_conditional_allocation_partial_capacity_branches(self):
        base = {
            "scenario_id": "s",
            "managed_flag": False,
            "vehicle_day_id": "vd",
            "home_record_id": "hh",
            "home_geoid": "360010001001",
            "home_citytown": "Albany",
            "trip_chain_sequence": 1,
            "ACTIVITY_TYPE": "WORK",
            "WHYTRP1S": 10,
            "managed_eligible_flag": False,
            "daily_energy_kwh": 10.0,
            "destination_lon": -73.7,
            "destination_lat": 42.7,
            "origin_lon": -73.75,
            "origin_lat": 42.65,
            "input_origin_lon": -73.75,
            "input_origin_lat": 42.65,
            "input_origin_mismatch_flag": False,
            "home_lon": -73.75,
            "home_lat": 42.65,
            "HHI": "MI",
            "HHS": 2,
            "VEH": 1,
            "URBRUR": 1,
            "HOMEOWN": 1,
        }
        stops = pd.DataFrame(
            [
                {
                    **base,
                    "TDTRPNUM": 1,
                    "charging_stop_id": "s1",
                    "charge_location_type": "work",
                    "charger_type": "work_l2",
                    "stop_priority": 1,
                    "rated_power_kw": 4.0,
                    "charger_availability_probability": 0.5,
                    "arrival_time_min": 500.0,
                    "departure_time_min": 560.0,
                    "dwell_min": 60.0,
                    "network_miles": 10.0,
                    "source_row_id": 1,
                },
                {
                    **base,
                    "TDTRPNUM": 2,
                    "charging_stop_id": "s2",
                    "charge_location_type": "quick_public",
                    "charger_type": "dcfc_low",
                    "stop_priority": 2,
                    "rated_power_kw": 10.0,
                    "charger_availability_probability": 1.0,
                    "arrival_time_min": 700.0,
                    "departure_time_min": 760.0,
                    "dwell_min": 60.0,
                    "network_miles": 0.0,
                    "source_row_id": 2,
                },
            ]
        )
        events, summary = bcm.build_charging_events(
            stops,
            scenario_id="partial_capacity",
            managed=False,
            efficiency_kwh_per_mile=1.0,
        )
        by_stop = dict(zip(events["charging_stop_id"], events["energy_delivered_kwh"]))
        self.assertAlmostEqual(by_stop["s1"], 2.0)
        self.assertAlmostEqual(by_stop["s2"], 8.0)
        self.assertAlmostEqual(summary["energy_delivered_kwh"].iloc[0], 10.0)

    def test_load_curve_wraps_to_next_day_bins(self):
        events = pd.DataFrame(
            [
                {
                    "scenario_id": "s",
                    "managed_flag": False,
                    "charger_type": "home_l2",
                    "charge_location_type": "home",
                    "home_geoid": "36001",
                    "start_time_min": 1430.0,
                    "end_time_min": 1450.0,
                    "expected_power_kw": 6.0,
                }
            ]
        )
        load = bcm.build_load_curve(events, location_columns=["home_geoid"], bin_minutes=15)
        by_bin = dict(zip(load["time_bin_min"], load["kwh"]))
        self.assertAlmostEqual(by_bin[1425], 1.0)
        self.assertAlmostEqual(by_bin[0], 1.0)

    def test_zcta_loader_accepts_2020_property_names(self):
        payload = {
            "type": "FeatureCollection",
            "features": [
                {
                    "type": "Feature",
                    "properties": {
                        "ZCTA5CE20": "12345",
                        "INTPTLON20": "-73.5",
                        "INTPTLAT20": "42.5",
                    },
                    "geometry": {
                        "type": "Polygon",
                        "coordinates": [
                            [
                                [-74.0, 42.0],
                                [-73.0, 42.0],
                                [-73.0, 43.0],
                                [-74.0, 43.0],
                                [-74.0, 42.0],
                            ]
                        ],
                    },
                }
            ],
        }
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "zcta.geojson"
            path.write_text(json.dumps(payload), encoding="utf-8")
            features = bcm.load_zcta_features(path, zip_to_county_path=None, county_filter=None)
        self.assertEqual(features[0]["zcta"], "12345")
        self.assertAlmostEqual(float(features[0]["centroid_lon"]), -73.5)

    def test_nearest_zcta_fallback_is_thresholded(self):
        features = [
            {
                "zcta": "12345",
                "geometry": {
                    "type": "Polygon",
                    "coordinates": [
                        [
                            [-74.0, 42.0],
                            [-73.0, 42.0],
                            [-73.0, 43.0],
                            [-74.0, 43.0],
                            [-74.0, 42.0],
                        ]
                    ],
                },
                "bbox": (-74.0, 42.0, -73.0, 43.0),
                "centroid_lon": -73.5,
                "centroid_lat": 42.5,
            }
        ]
        points = pd.DataFrame({"lon": [-80.0], "lat": [42.5]})
        no_fallback = bcm.assign_zcta_for_unique_points(
            points,
            lon_col="lon",
            lat_col="lat",
            zcta_features=features,
            fallback_nearest=False,
            nearest_max_miles=2.0,
        )
        thresholded = bcm.assign_zcta_for_unique_points(
            points,
            lon_col="lon",
            lat_col="lat",
            zcta_features=features,
            fallback_nearest=True,
            nearest_max_miles=2.0,
        )
        self.assertTrue(pd.isna(no_fallback["zcta"].iloc[0]))
        self.assertTrue(pd.isna(thresholded["zcta"].iloc[0]))
        self.assertGreater(thresholded["zcta_nearest_distance_miles"].iloc[0], 2.0)

    def test_h3_uses_raw_destination_coordinates_not_rounded_point(self):
        original = bcm._h3_latlng_to_cell
        try:
            bcm._h3_latlng_to_cell = lambda lat, lon, resolution: f"{lat:.5f}_{lon:.5f}_{resolution}"
            events = pd.DataFrame(
                [
                    {
                        "destination_lat": 42.123456,
                        "destination_lon": -73.654321,
                    }
                ]
            )
            out, summary = bcm.add_charging_spatial_ids(
                events,
                point_precision=2,
                h3_resolution=8,
                require_h3=True,
            )
        finally:
            bcm._h3_latlng_to_cell = original
        self.assertEqual(out["charging_lat"].iloc[0], 42.12)
        self.assertEqual(out["charging_lon"].iloc[0], -73.65)
        self.assertEqual(out["charging_h3"].iloc[0], "42.12346_-73.65432_8")
        self.assertEqual(summary["unique_charging_h3"], 1)

    def test_invalid_charger_assumptions_raise_error(self):
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "chargers.csv"
            pd.DataFrame(
                [
                    {
                        "charge_location_type": "home",
                        "charger_type": "home_l2",
                        "rated_power_kw": 7.2,
                        "availability_probability": 1.2,
                        "managed_eligible_flag": True,
                    }
                ]
            ).to_csv(path, index=False)
            with self.assertRaisesRegex(ValueError, "probability"):
                bcm.load_charger_assumptions(path)

    def test_adoption_duplicate_rows_raise_error(self):
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "adoption.csv"
            pd.DataFrame(
                [
                    {
                        "adoption_scenario": "base",
                        "forecast_year": 2030,
                        "home_zcta": "12203",
                        "adoption_fraction": 0.1,
                    },
                    {
                        "adoption_scenario": "base",
                        "forecast_year": 2030,
                        "home_zcta": "12203",
                        "adoption_fraction": 0.2,
                    },
                ]
            ).to_csv(path, index=False)
            with self.assertRaisesRegex(ValueError, "Duplicate"):
                bcm.load_adoption_file(path)

    def test_adoption_missing_home_zcta_raises_error(self):
        load = pd.DataFrame(
            [
                {
                    "home_zcta": "12203",
                    "scenario_id": "s",
                    "managed_flag": False,
                    "charger_type": "home_l2",
                    "charge_location_type": "home",
                    "time_bin_min": 0,
                    "time_bin_label": "00:00",
                    "kw": 1.0,
                    "kwh": 0.25,
                }
            ]
        )
        adoption = pd.DataFrame(
            [
                {
                    "adoption_scenario": "base",
                    "forecast_year": 2030,
                    "home_zcta": "99999",
                    "adoption_fraction": 0.1,
                    "vehicle_growth_factor": 1.0,
                }
            ]
        )
        with self.assertRaisesRegex(ValueError, "does not cover every modeled home_zcta"):
            bcm.apply_adoption_scaling(load, adoption, location_columns=["charging_h3"])

    def test_adoption_missing_home_zcta_in_one_scenario_year_raises_error(self):
        load = pd.DataFrame(
            [
                {
                    "home_zcta": "12203",
                    "scenario_id": "s",
                    "managed_flag": False,
                    "charger_type": "home_l2",
                    "charge_location_type": "home",
                    "charging_h3": "h1",
                    "time_bin_min": 0,
                    "time_bin_label": "00:00",
                    "kw": 1.0,
                    "kwh": 0.25,
                },
                {
                    "home_zcta": "12205",
                    "scenario_id": "s",
                    "managed_flag": False,
                    "charger_type": "home_l2",
                    "charge_location_type": "home",
                    "charging_h3": "h1",
                    "time_bin_min": 0,
                    "time_bin_label": "00:00",
                    "kw": 1.0,
                    "kwh": 0.25,
                },
            ]
        )
        adoption = pd.DataFrame(
            [
                {
                    "adoption_scenario": "base",
                    "forecast_year": 2030,
                    "home_zcta": "12203",
                    "adoption_fraction": 0.1,
                    "vehicle_growth_factor": 1.0,
                },
                {
                    "adoption_scenario": "base",
                    "forecast_year": 2030,
                    "home_zcta": "12205",
                    "adoption_fraction": 0.1,
                    "vehicle_growth_factor": 1.0,
                },
                {
                    "adoption_scenario": "base",
                    "forecast_year": 2035,
                    "home_zcta": "12203",
                    "adoption_fraction": 0.2,
                    "vehicle_growth_factor": 1.0,
                },
            ]
        )
        with self.assertRaisesRegex(ValueError, "First incomplete group"):
            bcm.apply_adoption_scaling(load, adoption, location_columns=["charging_h3"])

    def test_scale_adoption_skip_h3_raises(self):
        result = subprocess.run(
            [
                sys.executable,
                str(ROOT / "scripts" / "build_behavioral_charging_model.py"),
                "--sample-rows",
                "1",
                "--scale-adoption",
                "--skip-h3",
                "--use-demo-adoption",
                "--output-dir",
                str(ROOT / "out_behavioral_charging_skip_h3_test"),
            ],
            cwd=ROOT,
            text=True,
            capture_output=True,
            check=False,
        )
        self.assertNotEqual(result.returncode, 0)
        self.assertIn("--scale-adoption requires charging H3 outputs", result.stderr)

    def test_h3_adoption_smoke_writes_expected_outputs(self):
        adoption = ROOT / "models" / "adoption_forecast_albany_zip_for_charging.csv"
        if not adoption.exists():
            self.skipTest("Prepared Albany charging adoption file is not present")
        with tempfile.TemporaryDirectory() as tmp:
            trip_input = Path(tmp) / "trip_fixture.csv"
            zcta_geojson = Path(tmp) / "zcta_fixture.geojson"
            zcta_geojson.write_text(
                json.dumps(
                    {
                        "type": "FeatureCollection",
                        "features": [
                            {
                                "type": "Feature",
                                "properties": {
                                    "ZCTA5CE20": "12203",
                                    "INTPTLON20": "-73.80",
                                    "INTPTLAT20": "42.68",
                                },
                                "geometry": {
                                    "type": "Polygon",
                                    "coordinates": [
                                        [
                                            [-73.90, 42.60],
                                            [-73.70, 42.60],
                                            [-73.70, 42.80],
                                            [-73.90, 42.80],
                                            [-73.90, 42.60],
                                        ]
                                    ],
                                },
                            }
                        ],
                    }
                ),
                encoding="utf-8",
            )
            pd.DataFrame(
                [
                    trip_row(
                        SYN_RECORD_ID="SMOKE00000001",
                        home_geoid="360010001001",
                        PERSONID=1,
                        TDTRPNUM=1,
                        STRTTIME=800,
                        ENDTIME=820,
                        TRVLCMIN=20,
                        TRPMILES=5.0,
                        DWELTIME=540,
                        ACTIVITY_TYPE="WORK",
                        ACTIVITY_X=-73.85,
                        ACTIVITY_Y=42.73,
                        START_X=-73.78,
                        START_Y=42.66,
                        HOME_X=-73.78,
                        HOME_Y=42.66,
                        SHORTEST_TDIST_miles=5.5,
                        SHORTEST_PDIST_miles=5.1,
                    ),
                    trip_row(
                        SYN_RECORD_ID="SMOKE00000001",
                        home_geoid="360010001001",
                        PERSONID=1,
                        TDTRPNUM=2,
                        STRTTIME=1700,
                        ENDTIME=1725,
                        TRVLCMIN=25,
                        TRPMILES=5.0,
                        DWELTIME=-9,
                        ACTIVITY_TYPE="HOME",
                        ACTIVITY_X=-73.78,
                        ACTIVITY_Y=42.66,
                        START_X=-73.85,
                        START_Y=42.73,
                        HOME_X=-73.78,
                        HOME_Y=42.66,
                        SHORTEST_TDIST_miles=5.5,
                        SHORTEST_PDIST_miles=5.1,
                    ),
                ]
            ).drop(columns=["source_row_id"]).to_csv(trip_input, index=False)
            result = subprocess.run(
                [
                    sys.executable,
                    str(ROOT / "scripts" / "build_behavioral_charging_model.py"),
                    "--trip-input",
                    str(trip_input),
                    "--sample-rows",
                    "25",
                    "--require-h3",
                    "--zcta-geojson",
                    str(zcta_geojson),
                    "--zcta-county-filter",
                    "",
                    "--scale-adoption",
                    "--adoption-file",
                    str(adoption),
                    "--charger-assumptions",
                    str(ROOT / "Albany_Trip_Behaviour" / "config" / "charger_assumptions.csv"),
                    "--output-dir",
                    tmp,
                ],
                cwd=ROOT,
                text=True,
                capture_output=True,
                check=False,
            )
            self.assertEqual(result.returncode, 0, result.stderr)
            for name in [
                "charging_events.csv",
                "full_ev_load_15min_by_home_zcta_charging_h3.csv",
                "scaled_load_15min_by_charging_h3.csv",
                "distance_qa.csv",
            ]:
                path = Path(tmp) / name
                self.assertTrue(path.exists(), name)
                self.assertGreater(path.stat().st_size, 0, name)


if __name__ == "__main__":
    unittest.main()
