$(document).ready(function () {
    const baseUrl = (window.location.hostname === "localhost" || window.location.hostname === "127.0.0.1")
    ? window.location.origin
    : "https://www.electforecast.de";

    console.log("Base URL:", baseUrl);

    // Update the "last updated" timestamps
    function updateLastUpdated() {
        $.ajax({
            url: `${baseUrl}/api/lastUpdated`,
            type: 'GET',
            success: function (response) {
                if (response.lastUpdatedDemand) {
                    $('#last-updated-demand em').text(`Data last updated as of ${response.lastUpdatedDemand}`);
                }
                if (response.lastUpdatedWeather) {
                    $('#last-updated-weather em').text(`Data last updated as of ${response.lastUpdatedWeather}`);
                }
                if (response.lastUpdatedHoliday) {
                    $('#last-updated-holiday em').text(`Data last updated as of ${response.lastUpdatedHoliday}`);
                }
            },
            error: function (xhr, status, error) {
                console.error('Failed to fetch last updated timestamp:', error);
                console.error("XHR Error:", xhr.responseText);

            }
        });
    }
      // Initialize the Hourly Demand Table
    function initializeHourlyDemandTable() {
        if ($.fn.DataTable.isDataTable('#hourly-demand-table')) {
            $('#hourly-demand-table').DataTable().destroy();
        }

        $('#hourly-demand-table').DataTable({
            processing: true,
            serverSide: true,
            ajax: {
                url: `${baseUrl}/api/hourlydemand`,
                type: 'GET',
                dataSrc: function (json) {
                    if (!json.data) {
                        console.error("Invalid Hourly Demand JSON response:", json);
                        return [];
                    }
                    console.log("API Response for Hourly Demand:", json);
                    return json.data;
                },
                error: function (xhr, error, code) {
                    console.error("Error fetching hourly demand data:", error);
                    console.error("XHR Error:", xhr.responseText);
                    alert("Failed to load demand data. Please try again later.");
                }
            },
            columns: [
                { data: 'time', title: 'Datetime' },
                { data: 'value', title: 'Demand (kWh)' }
            ],
            order: [[0, 'desc']],
            pageLength: 10,
        });
    }

    // Initialize the Hourly Weather Table
    function initializeHourlyWeatherTable() {
        if ($.fn.DataTable.isDataTable('#hourly-weather-table')) {
            $('#hourly-weather-table').DataTable().destroy();
        }

        $('#hourly-weather-table').DataTable({
            processing: true,
            serverSide: true,
            ajax: {
                url: `${baseUrl}/api/hourlyweather`,
                type: 'GET',
                headers: {
                    'Content-Type': 'application/json',
                },
                dataSrc: function (json) {
                    if (!json.data) {
                        console.error("Invalid Hourly Weather JSON response:", json);
                        return [];
                    }
                    console.log("API Response for Hourly Weather:", json);
                    return json.data;
                },
                error: function (xhr, error, code) {
                    console.error("Error fetching hourly weather data:", error);
                    console.error("XHR Error:", xhr.responseText);
                    alert("Failed to load weather data. Please try again later.");
                }
            },
            columns: [
                { data: 'datetime', title: 'Datetime' },
                { data: 'temp', title: 'Temperature (°C)' },
                { data: 'feelslike', title: 'Feels Like (°C)' },
                { data: 'humidity', title: 'Humidity (%)' },
                { data: 'windspeed', title: 'Wind Speed (km/h)' },
                { data: 'cloudcover', title: 'Cloud Cover (%)' },
                { data: 'solaradiation', title: 'Solar Radiation (W/m²)' },
                { data: 'precip', title: 'Precipitation (mm)' },
                { data: 'preciptype', title: 'Precipitation Type' }
            ],
            order: [[0, 'desc']],
            pageLength: 10,
        });
    }

    // Initialize the Holidays Table
    function initializeHolidaysTable() {
        if ($.fn.DataTable.isDataTable('#holidays-table')) {
            $('#holidays-table').DataTable().destroy();
        }

        $('#holidays-table').DataTable({
            serverSide: true,
            processing: true,
            ajax: {
                url: `${baseUrl}/api/holidays`,
                type: 'GET',
                headers: {
                    'Content-Type': 'application/json',
                },
                dataSrc: function (json) {
                    if (!json.data) {
                        console.error("Invalid Holidays JSON response:", json);
                        return [];
                    }
                    console.log("API Response for Holidays:", json);
                    return json.data;
                },
                error: function (xhr, error, code) {
                    console.error("Error fetching holidays data:", error);
                    console.error("XHR Error:", xhr.responseText);
                    alert("Failed to load holidays. Please try again later.");
                }
            },
            columns: [
                { data: 'date', title: 'Date' },
                { data: 'name', title: 'Holiday Name' }
            ],
            order: [[0, 'desc']],
            pageLength: 10,
        });
    }

    updateLastUpdated();
    setInterval(updateLastUpdated, 60000);

    // Initialize all tables
    initializeHourlyDemandTable();
    initializeHourlyWeatherTable();
    initializeHolidaysTable();

    // Search functionality (applies to all tables)
    let debounceTimer;
    $('#searchInput').on('input', function () {
        clearTimeout(debounceTimer);
        const searchValue = $(this).val();

        debounceTimer = setTimeout(function () {
            $('#hourly-demand-table').DataTable().search(searchValue).draw();
            $('#hourly-weather-table').DataTable().search(searchValue).draw();
        }, 300);
    });

    // Sidebar menu search
    $('#menuSearch').on('input', function () {
        const query = $(this).val().toLowerCase();
        $('.sidebar .nav-item').each(function () {
            const text = $(this).text().toLowerCase();
            if (text.includes(query)) {
                $(this).show();
            } else {
                $(this).hide();
            }
        });
    });
});
