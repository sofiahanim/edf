$(document).ready(function () {
    const baseUrl = window.location.origin;

    // Initialize Hourly Demand Table
    function initializeHourlyDemandTable() {
        if ($.fn.DataTable.isDataTable('#hourly-demand-table')) {
            $('#hourly-demand-table').DataTable().destroy(); // Prevent reinitialization
        }

        $('#hourly-demand-table').DataTable({
            processing: true,  // Show processing indicator
            serverSide: true,  // Enable server-side processing
            ajax: {
                url: `${baseUrl}/api/hourlydemand`,
                type: 'GET',
                dataSrc: function (json) {
                    console.log("API Response for Hourly Demand:", json);  // Debugging API response
                    return json.data;  // Pass the data to DataTables
                },
            },
            columns: [
                { data: 'time', title: 'Datetime' },
                { data: 'value', title: 'Demand (kWh)' },
            ],
            order: [[0, 'desc']],  // Sort by latest timestamp
            pageLength: 10,        // Display 10 records per page
        });
    }

    // Initialize Hourly Weather Table
    function initializeHourlyWeatherTable() {
        if ($.fn.DataTable.isDataTable('#hourly-weather-table')) {
            $('#hourly-weather-table').DataTable().destroy(); // Prevent reinitialization
        }

        $('#hourly-weather-table').DataTable({
            processing: true,  // Show processing indicator
            serverSide: true,  // Enable server-side processing
            ajax: {
                url: `${baseUrl}/api/hourlyweather`,  // API endpoint
                type: 'GET',
            },
            columns: [
                { data: 'datetime', title: 'Datetime' },
                { data: 'temp', title: 'Temperature (°C)' },
                { data: 'feelslike', title: 'Feels Like (°C)' },
                { data: 'humidity', title: 'Humidity (%)' },
                { data: 'windspeed', title: 'Wind Speed (km/h)' },
                { data: 'cloudcover', title: 'Cloud Cover (%)' },
                { data: 'solarradiation', title: 'Solar Radiation (W/m²)' },
                { data: 'precip', title: 'Precipitation (mm)' },
                { data: 'preciptype', title: 'Precipitation Type' },
            ],
            order: [[0, 'desc']],  // Sort by latest `datetime` by default
            pageLength: 10,        // Display 10 records per page
        });
    }

    // Initialize Holidays Table
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
                dataSrc: 'data',
            },
            columns: [
                { data: 'date', title: 'Date' },
                { data: 'name', title: 'Holiday Name' }
            ],
            order: [[0, 'desc']],
            language: {
                emptyTable: "No holidays available",
                processing: "Loading holidays, please wait...",
            },
            pageLength: 10,
        });
    }

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
