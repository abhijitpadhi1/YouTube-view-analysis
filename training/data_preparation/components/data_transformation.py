import os, sys
import pandas as pd
import numpy as np

# Import custom modules
from training.logging.logger import Logger
from training.exception.exception import CustomException
# Import entity and constants
from training.data_preparation.entity.config_entity import DataTransformationConfig
from training.data_preparation.entity.artifact_entity import DataValidationArtifact, DataTransformationArtifact


class DataTransformation:
    def __init__(
        self, 
        data_transformation_config: DataTransformationConfig, 
        data_validation_artifact: DataValidationArtifact
    ):
        try:
            self.data_transformation_config = data_transformation_config
            self.data_validation_artifact = data_validation_artifact
        except Exception as e:
            err = CustomException(str(e), sys)
            Logger().log(f"Error in DataTransformation __init__: {err}", level="error")
            raise err

    @staticmethod
    def read_data(file_path: str) -> pd.DataFrame:
        try:
            Logger().log(f"Reading data from file: {file_path}")
            return pd.read_parquet(file_path)
        except Exception as e:
            err = CustomException(str(e), sys)
            Logger().log(f"Error in read_data: {err}", level="error")
            raise err

    def apply_datetime_normalization(self, df: pd.DataFrame) -> pd.DataFrame:
        """Apply datetime normalization and feature extraction to the DataFrame."""
        try:
            Logger().log("Starting datetime normalization.")

            # Convert 'video_published_at' to datetime and extract features
            df['video_published_at'] = pd.to_datetime(df['video_published_at'], utc=True)
            # Extracting publish hour, day of week, and whether it's a weekend
            df['publish_hour'] = df['video_published_at'].dt.hour # type: ignore
            df['publish_day_of_week'] = df['video_published_at'].dt.day # type: ignore
            df['is_weekend'] = df['publish_day_of_week'].apply(lambda x: 1 if x >= 5 else 0)

            ## From both publish & trending:
            # Convert 'video_trending__date' to datetime
            df['video_trending__date'] = pd.to_datetime(df['video_trending__date'], utc=True)
            # hours_to_trend = trending_date − publish_date
            df['hours_to_trend'] = (df['video_trending__date'] - df['video_published_at']).dt.total_seconds() # type: ignore
            # For negative values make them to zero
            df['hours_to_trend'] = df['hours_to_trend'].apply(lambda x: 0 if x < 0 else x)
            df['hours_to_trend'] = df['hours_to_trend'] / 3600

            ## From channel:
            # Convert 'channel_published_at' to datetime
            df['channel_published_at'] = pd.to_datetime(df['channel_published_at'], format='ISO8601', utc=True)
            # channel_age_days = trending_date − channel_published_at
            df['channel_age_days'] = (df['video_trending__date'] - df['channel_published_at']).dt.days # type: ignore

            return df
        except Exception as e:
            err = CustomException(str(e), sys)
            Logger().log(f"Error in datetime_normalization: {err}", level="error")
            raise err

    def apply_duration_transformation(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Apply duration transformation to the DataFrame by creating duration buckets.

        Bucket Rules to be applied on 'video_duration' column:
            short   : < 180 sec
            medium  : 180-600 sec
            long    : > 600 sec        
        """
        try:
            Logger().log("Starting duration transformation.")

            # Define bins and labels for duration bucketing
            bins = [0, 180, 600, float('inf')]
            labels = ['short', 'medium', 'long']

            # Convert 'video_duration' to timedelta and then to total seconds
            df['video_duration'] = pd.to_timedelta(df['video_duration'])
            df['duration_seconds'] = df['video_duration'].dt.total_seconds() # type: ignore
            # Create duration buckets based on the defined bins and labels
            df['duration_bucket'] = pd.cut(df['duration_seconds'], bins=bins, labels=labels, include_lowest=True)    

            return df
        except Exception as e:
            err = CustomException(str(e), sys)
            Logger().log(f"Error in duration_transformation: {err}", level="error")
            raise err

    def apply_engagement_ratio_calculation(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Apply engagement ratio calculation to the DataFrame by creating new features for like, dislike, comment, and view ratios.

        Engagement Ratio Calculation:
            like_ratio = video_likes / video_views * 1000
            comment_ratio = video_comments / video_views * 1000
            engagement_score = likes + 2 * comments
        """
        try:
            Logger().log("Starting engagement ratio calculation.")
            # Calculate engagenent ratio
            df['likes_per_1k_views'] = (df['video_like_count'] / df['video_view_count']) * 1000
            df['comments_per_1k_views'] = (df['video_comment_count'] / df['video_view_count']) * 1000
            df['engagement_score'] = df['video_like_count'] + 2 * df['video_comment_count']

            return df
        except Exception as e:
            err = CustomException(str(e), sys)
            Logger().log(f"Error in engagement_ratio_calculation: {err}", level="error")
            raise err

    def apply_channel_popularity_transformation(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Apply channel popularity transformation to the DataFrame by creating channel subscriber count buckets.
        
            Subscriber Count Bucketing:
                nano   (<10K)
                micro  (10K-100K)
                mid    (100K-1M)
                macro  (>1M)
        """
        try:
            Logger().log("Starting channel popularity transformation.")
            # Define bins and labels for channel subscriber count bucketing
            bins = [0, 1e+4, 1e+5, 1e+6, float('inf')]
            labels = ['nano', 'micro', 'mid', 'macro']
            # Create channel subscriber count buckets based on the defined bins and labels
            df['channel_subscriber_bucket'] = pd.cut(df['channel_subscriber_count'], bins=bins, labels=labels, include_lowest=True)

            return df
        except Exception as e:
            err = CustomException(str(e), sys)
            Logger().log(f"Error in channel_popularity_transformation: {err}", level="error")
            raise err

    def apply_geographic_harmonization(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Apply geographic harmonization to the DataFrame by creating a flag for whether the video is trending in the same country as the channel's country.
            
            Create 'same_country_flag' if video_trending_country == channel_country
        """
        try:
            # Create 'same_country_flag' based on whether video_trending_country is the same as channel_country
            Logger().log("Starting geographic harmonization.")
            df['same_country_flag'] = (df['video_trending_country'] == df['channel_country']).astype(int)
            return df
        except Exception as e:
            err = CustomException(str(e), sys)
            Logger().log(f"Error in geographic_harmonization: {err}", level="error")
            raise err

    def apply_feature_engineering(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Apply all feature engineering steps to the DataFrame in a sequential manner.
        """
        try:
            Logger().log("Starting feature engineering process.")
            df = self.apply_datetime_normalization(df)
            df = self.apply_duration_transformation(df)
            df = self.apply_engagement_ratio_calculation(df)
            df = self.apply_channel_popularity_transformation(df)
            df = self.apply_geographic_harmonization(df)

            Logger().log("Feature engineering process completed successfully.")
            return df
        except Exception as e:
            err = CustomException(str(e), sys)
            Logger().log(f"Error in apply_feature_engineering: {err}", level="error")
            raise err
        
    def assert_sanity_checks(self, df: pd.DataFrame) -> None:
        try:
            Logger().log("Starting sanity checks on the transformed DataFrame.")
            # Assert no negative durations
            assert (df['duration_seconds'] >= 0).all(), "Negative durations found!"

            # Assert hours_to_trend >= 0
            assert (df['hours_to_trend'] >= 0).all(), "Negative hours_to_trend found!"

            # Assert no infinite values in the engagement ratios
            assert not np.isinf(df[['likes_per_1k_views','comments_per_1k_views']]).any().any()

            # Check the percentage of missing values in the original engagement signals and the derived ratios
            sanity_report = df[['likes_missing_flag','comments_missing_flag','views_invalid_flag']].mean()
            
            # `views_invalid_flag` should ideally be 0% if all invalid view counts were dropped, while 
            # `likes_missing_flag` and `comments_missing_flag` can be >0% due to missing values that were imputed.
            assert df['views_invalid_flag'].sum() == 0
            assert df['likes_missing_flag'].sum() > 0
            assert df['comments_missing_flag'].sum() > 0

            # print("All assertions correctly passed. Sanity checks successful!")
            Logger().log(f"Sanity checks passed successfully. {sanity_report.to_dict()}")

        except Exception as e:
            err = CustomException(str(e), sys)
            Logger().log(f"Error in assert_sanity_checks: {err}", level="error")
            raise err

    def handle_missing_values(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Handle missing values in the DataFrame by applying appropriate imputation strategies and creating missingness flags.
        """
        try:
            Logger().log("Starting missing value handling.")
            # Drop rows where 'video_view_count' is null
            df = df.dropna(subset=['video_view_count'])
            # If null 'video_category_id' => 'unknown'
            df['video_category_id'] = df['video_category_id'].fillna('unknown')
            # If null channel_view_count => median (grouped by country)
            df['channel_view_count'] = df.groupby('channel_country')['channel_view_count'].transform(lambda x: x.fillna(x.median()))

            ## 1. Missingness / validity flags
            # Create missingness flags => likes_missing_flag, comments_missing_flag
            df['likes_missing_flag'] = df['video_like_count'].isna().astype(int)
            df['comments_missing_flag'] = df['video_comment_count'].isna().astype(int)
            df['views_invalid_flag'] = (df['video_view_count'].isna() | (df['video_view_count'] <= 0)).astype(int)

            # # Recreate the derieved features
            # if likes_missing_flag == 0:
            #     likes_per_1k_views = (likes / views) * 1000
            # else:
            #     likes_per_1k_views = NaN

            ## 2. Valid view condition
            valid_views = df['views_invalid_flag'] == 0

            ## 3. Derived ratios (ONLY when semantically valid)
            # If views are missing/invalid or likes/comments are missing, then the ratios should be set to NaN to avoid misleading values.
            df['likes_per_1k_views'] = np.where(
                (df['likes_missing_flag'] == 0) & valid_views,
                (df['video_like_count'] / df['video_view_count']) * 1000,
                np.nan
            )
            # If views are missing/invalid or comments are missing, then the ratios should be set to NaN to avoid misleading values.
            df['comments_per_1k_views'] = np.where(
                (df['comments_missing_flag'] == 0) & valid_views,
                (df['video_comment_count'] / df['video_view_count']) * 1000,
                np.nan
            )

            # 4. Engagement score (only if BOTH engagement signals exist)
            df['engagement_score'] = np.where(
                (df['likes_missing_flag'] == 0) & (df['comments_missing_flag'] == 0),
                df['video_like_count'] + 2 * df['video_comment_count'],
                np.nan
            )

            return df
        except Exception as e:
            err = CustomException(str(e), sys)
            Logger().log(f"Error in handle_missing_values: {err}", level="error")
            raise err

    def initiate_data_transformation(self) -> DataTransformationArtifact:
        try:
            Logger().log("Starting data transformation process.")
            # Read the validated data
            validated_data_path = self.data_validation_artifact.validated_data_path
            df = self.read_data(validated_data_path)
            Logger().log(f"Data read successfully from {validated_data_path}. Starting transformation.")

            # Apply feature engineering and data transformations
            df = self.apply_feature_engineering(df)
            # Handle missing values first before applying feature engineering
            df = self.handle_missing_values(df)
            # Assert sanity checks on the transformed DataFrame
            self.assert_sanity_checks(df)

            # Save the transformed data to the specified directory
            transformed_data_dir = self.data_transformation_config.transformed_data_dir
            os.makedirs(transformed_data_dir, exist_ok=True)
            transformed_data_path = self.data_transformation_config.transformed_data_file_path
            df.to_parquet(transformed_data_path, index=False)
            Logger().log(f"Transformed data saved successfully at {transformed_data_path}.")

            # Create and return the DataTransformationArtifact
            data_transformation_artifact = DataTransformationArtifact(
                transformed_data_dir=transformed_data_dir,
                transformed_data_file_path=transformed_data_path
            )
            Logger().log(f"Data transformation artifact created successfully: {data_transformation_artifact}")
            return data_transformation_artifact

        except Exception as e:
            err = CustomException(str(e), sys)
            Logger().log(f"Error in initiate_data_transformation: {err}", level="error")
            raise err

        
            